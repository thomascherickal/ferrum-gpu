//! Summarising a path: what it is, and for a folder how many files and bytes lie under it.

use super::{MAX_DEPTH, MAX_WALK_FILES};
use std::path::Path;

/// What a path is and how big.
#[derive(Debug, PartialEq)]
pub struct Stat {
    /// `file`, `dir` or `missing`.
    pub kind: &'static str,
    /// Bytes of a file, or the total of the files found under a folder.
    pub bytes: u64,
    /// Files found under a folder (1 for a file).
    pub files: usize,
    /// The walk stopped at a limit, so `files` and `bytes` are lower bounds.
    pub truncated: bool,
}

pub(super) fn walk(dir: &Path, depth: usize, limits: (usize, usize), s: &mut Stat) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for item in rd.flatten() {
        if s.files >= limits.0 {
            s.truncated = true;
            return;
        }
        let Ok(meta) = std::fs::metadata(item.path()) else {
            continue;
        };
        if meta.is_file() {
            s.files += 1;
            s.bytes += meta.len();
        }
        if meta.is_dir() {
            if depth >= limits.1 {
                s.truncated = true;
            } else {
                walk(&item.path(), depth + 1, limits, s);
            }
        }
    }
}

/// Summarise `path`; a missing path is a `missing` result, not an error.
pub fn stat(path: &Path) -> Stat {
    stat_with(path, (MAX_WALK_FILES, MAX_DEPTH))
}

/// [`stat`] with explicit `(max files, max depth)` limits.
pub fn stat_with(path: &Path, limits: (usize, usize)) -> Stat {
    let mk = |kind, bytes, files| Stat {
        kind,
        bytes,
        files,
        truncated: false,
    };
    match std::fs::metadata(path) {
        Err(_) => mk("missing", 0, 0),
        Ok(m) if m.is_file() => mk("file", m.len(), 1),
        Ok(_) => {
            let mut s = mk("dir", 0, 0);
            walk(path, 0, limits, &mut s);
            s
        }
    }
}
