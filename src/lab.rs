use std::fs::{create_dir_all, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Mutex;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

/// One logged event. `Copy`/alloc-free so the hot path neither formats nor
/// allocates — it only timestamps and moves this into the channel.
#[derive(Clone, Copy, Debug)]
struct LogRow {
    unix_nanos: u128,
    event: &'static str,
    conn_id: u64,
    seq: u64,
    bytes: usize,
    kind: &'static str,
}

/// Lab-only traffic logger.
///
/// IMPORTANT — measurement must not perturb the thing it measures. The shaped
/// wire emits cells at a high fixed rate across many flows; if `log()` did a
/// synchronous, mutex-guarded, unbuffered file write on the hot path it would
/// throttle and serialise the emitter (the shared emitter thread would block on
/// file I/O ~tens-of-thousands of times a second, and that contention is itself
/// activity-correlated), distorting the recorded timing and inventing a leak
/// that does not exist on the real wire. So the hot path does the MINIMUM — grab
/// a timestamp and move an alloc-free `LogRow` into a channel; ALL formatting and
/// buffered I/O happen on a dedicated writer thread. (An earlier version still
/// did the row `format!` — and its per-cell heap alloc — on the hot path, which
/// left a faint activity-correlated allocator-contention jitter; moving the
/// format off-thread removes that too.) Timestamps are taken at call time, so
/// timing fidelity is preserved even though writes are deferred and may be
/// reordered (the analyzer sorts by timestamp).
#[derive(Debug)]
pub struct LabLogger {
    tx: Option<Sender<LogRow>>,
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

        let (tx, rx) = mpsc::channel::<LogRow>();
        // Background writer: formats each row and writes it with a buffered
        // writer, flushing periodically so a reader sees data promptly without
        // the hot path paying for either formatting or syscalls.
        let handle = std::thread::Builder::new()
            .name(format!("lab-log-{role}"))
            .spawn(move || {
                let mut since_flush = 0usize;
                while let Ok(r) = rx.recv() {
                    if writeln!(
                        writer,
                        "{},{role},{},{},{},{},{}",
                        r.unix_nanos, r.event, r.conn_id, r.seq, r.bytes, r.kind
                    )
                    .is_err()
                    {
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
        // Hot path does the MINIMUM: timestamp + move an alloc-free row into the
        // channel. No format!, no heap alloc, no lock on a shared file.
        let unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();

        if let Some(tx) = &self.tx {
            let _ = tx.send(LogRow {
                unix_nanos,
                event,
                conn_id,
                seq,
                bytes,
                kind,
            });
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
