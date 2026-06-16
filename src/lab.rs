use std::fs::{create_dir_all, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Mutex;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

/// Lab-only traffic logger.
///
/// IMPORTANT — measurement must not perturb the thing it measures. The shaped
/// wire emits cells at a high fixed rate across many flows; if `log()` did a
/// synchronous, mutex-guarded, unbuffered file write on the hot path it would
/// throttle and serialise the emitter (the shared emitter thread would block on
/// file I/O ~tens-of-thousands of times a second, and that contention is itself
/// activity-correlated), distorting the recorded timing and inventing a leak
/// that does not exist on the real wire. So the hot path only captures the
/// timestamp and hands a preformatted row to a dedicated writer thread over a
/// lock-free-ish channel; all formatting cost is the caller's but all blocking
/// I/O happens off the hot path. Timestamps are taken at call time, so timing
/// fidelity is preserved even though writes are deferred and may be reordered
/// (the analyzer sorts by timestamp).
#[derive(Debug)]
pub struct LabLogger {
    role: &'static str,
    tx: Option<Sender<String>>,
    next_conn_id: AtomicU64,
    writer: Mutex<Option<JoinHandle<()>>>,
}

impl LabLogger {
    pub fn create(dir: impl AsRef<Path>, role: &'static str) -> io::Result<Arc<Self>> {
        create_dir_all(dir.as_ref())?;
        let path = dir.as_ref().join(format!("{role}.csv"));
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        let mut writer = BufWriter::with_capacity(1 << 16, file);
        writeln!(writer, "unix_nanos,role,event,conn_id,seq,bytes,kind")?;

        let (tx, rx) = mpsc::channel::<String>();
        // Background writer: drains rows and writes them with a buffered writer,
        // flushing periodically so a reader sees data promptly without making the
        // hot path pay for syscalls.
        let handle = std::thread::Builder::new()
            .name(format!("lab-log-{role}"))
            .spawn(move || {
                let mut since_flush = 0usize;
                while let Ok(row) = rx.recv() {
                    if writer.write_all(row.as_bytes()).is_err() {
                        break;
                    }
                    since_flush += 1;
                    if since_flush >= 1024 {
                        let _ = writer.flush();
                        since_flush = 0;
                    }
                }
                let _ = writer.flush();
            })?;

        Ok(Arc::new(Self {
            role,
            tx: Some(tx),
            next_conn_id: AtomicU64::new(1),
            writer: Mutex::new(Some(handle)),
        }))
    }

    pub fn next_conn_id(&self) -> u64 {
        self.next_conn_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn log(
        &self,
        event: &'static str,
        conn_id: u64,
        seq: u64,
        bytes: usize,
        kind: &'static str,
    ) {
        // Capture the timestamp on the hot path; defer formatting cost is small
        // and all blocking I/O happens on the writer thread.
        let unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();

        if let Some(tx) = &self.tx {
            let _ = tx.send(format!(
                "{unix_nanos},{},{event},{conn_id},{seq},{bytes},{kind}\n",
                self.role
            ));
        }
    }
}

impl Drop for LabLogger {
    fn drop(&mut self) {
        // Close the channel so the writer thread drains, flushes, and exits, then
        // join it so the CSV is complete before the process inspects it.
        self.tx = None;
        if let Some(handle) = self.writer.lock().ok().and_then(|mut w| w.take()) {
            let _ = handle.join();
        }
    }
}
