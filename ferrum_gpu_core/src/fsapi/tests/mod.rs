//! Tests of the file API, on real temp-directory trees.

mod list;
mod preview;
mod stat;

pub(crate) mod support {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};

    static N: AtomicU32 = AtomicU32::new(0);

    /// A fresh empty directory under the system temp directory.
    pub(crate) fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ferrum_fs_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Write `bytes` to `dir/rel`, creating folders.
    pub(crate) fn put(dir: &Path, rel: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, bytes).unwrap();
        p
    }
}
