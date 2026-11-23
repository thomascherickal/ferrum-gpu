//! A preview of the start of a text file.

use super::{MAX_COUNT_BYTES, MAX_PREVIEW};
use std::io::{self, Read};
use std::path::Path;

/// The start of a file.
#[derive(Debug, PartialEq)]
pub struct Preview {
    /// The first bytes as text (invalid UTF-8 shown as the replacement character).
    pub text: String,
    /// The size of the whole file.
    pub size: u64,
    /// Lines in the first [`MAX_COUNT_BYTES`] bytes.
    pub lines: u64,
    /// The text is shorter than the file.
    pub truncated: bool,
    /// The file has NUL bytes in its start, so it is probably not text.
    pub binary: bool,
}

/// Read up to `bytes` (at most [`MAX_PREVIEW`]) from the start of the file at `path`.
pub fn preview(path: &Path, bytes: usize) -> Result<Preview, String> {
    preview_with(path, bytes, &|p| Ok(Box::new(std::fs::File::open(p)?)))
}

/// How a file is opened for reading.
pub(super) type Opener<'a> = &'a dyn Fn(&Path) -> io::Result<Box<dyn Read>>;

/// [`preview`] reading through whatever `open` returns, so a failing source can be injected.
pub(super) fn preview_with(path: &Path, bytes: usize, open: Opener) -> Result<Preview, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{}: not a file", path.display()));
    }
    let want = bytes.clamp(1, MAX_PREVIEW);
    let mut f = open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut head = vec![0u8; want];
    let mut got = 0;
    while got < want {
        match f.read(&mut head[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
    }
    head.truncate(got);
    let binary = head.contains(&0);
    // count lines over a larger window than the preview, in chunks
    let mut lines = head.iter().filter(|&&b| b == b'\n').count() as u64;
    let mut seen = got as u64;
    let mut buf = vec![0u8; 64 * 1024];
    while seen < MAX_COUNT_BYTES {
        match f.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                lines += buf[..n].iter().filter(|&&b| b == b'\n').count() as u64;
                seen += n as u64;
            }
        }
    }
    Ok(Preview {
        text: String::from_utf8_lossy(&head).into_owned(),
        size: meta.len(),
        lines,
        truncated: meta.len() > got as u64,
        binary,
    })
}
