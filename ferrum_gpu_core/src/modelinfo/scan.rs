//! Finding the GGUF files in a folder, for the Models tab.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Most files a scan returns.
pub const MAX_FILES: usize = 500;
/// Deepest folder level a scan goes into (0 is the folder itself).
pub const MAX_DEPTH: usize = 3;

/// One model file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFile {
    /// Its path.
    pub path: PathBuf,
    /// Its size in bytes.
    pub size: u64,
    /// Last modified, in seconds since the epoch, if the file system says.
    pub modified: Option<u64>,
}

fn is_gguf(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

pub(super) fn walk(
    dir: &Path,
    depth: usize,
    max: usize,
    out: &mut Vec<ModelFile>,
    truncated: &mut bool,
) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let Ok(meta) = std::fs::metadata(e.path()) else {
            continue;
        };
        if meta.is_file() && is_gguf(&e.path()) {
            if out.len() >= max {
                *truncated = true;
                return;
            }
            out.push(ModelFile {
                path: e.path(),
                size: meta.len(),
                modified: meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs()),
            });
        } else if meta.is_dir()
            && depth < MAX_DEPTH
            && !e.file_name().to_string_lossy().starts_with('.')
        {
            walk(&e.path(), depth + 1, max, out, truncated);
        }
    }
}

/// The `.gguf` files under `dir` (hidden folders skipped), sorted by path within each folder, and whether
/// the list was cut at [`MAX_FILES`]. A folder that cannot be read is an error.
pub fn scan(dir: &Path) -> Result<(Vec<ModelFile>, bool), String> {
    scan_with(dir, MAX_FILES)
}

/// [`scan`] returning at most `max` files.
pub fn scan_with(dir: &Path, max: usize) -> Result<(Vec<ModelFile>, bool), String> {
    std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (mut out, mut truncated) = (vec![], false);
    walk(dir, 0, max, &mut out, &mut truncated);
    Ok((out, truncated))
}
