use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct LabLogger {
    role: &'static str,
    file: Mutex<File>,
    next_conn_id: AtomicU64,
}

impl LabLogger {
    pub fn create(dir: impl AsRef<Path>, role: &'static str) -> io::Result<Arc<Self>> {
        create_dir_all(dir.as_ref())?;
        let path = dir.as_ref().join(format!("{role}.csv"));
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        writeln!(file, "unix_nanos,role,event,conn_id,seq,bytes,kind")?;

        Ok(Arc::new(Self {
            role,
            file: Mutex::new(file),
            next_conn_id: AtomicU64::new(1),
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
        let unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();

        if let Ok(mut file) = self.file.lock() {
            let _ = writeln!(
                file,
                "{unix_nanos},{},{event},{conn_id},{seq},{bytes},{kind}",
                self.role
            );
        }
    }
}
