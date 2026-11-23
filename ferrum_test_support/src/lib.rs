//! Helpers the tests of the other crates share: failures and fixtures that exist on every
//! platform (Linux, macOS, Windows), so no test needs a `/proc` path or a Unix-only trick.
#![forbid(unsafe_code)]

use std::io::{self, Read};
use std::path::Path;

mod deadlined;
mod fake_tool;
mod guarded;
mod helper;
pub use deadlined::{with_deadline, Deadlined};
pub use fake_tool::FakeTool;
pub use guarded::Guarded;
pub use helper::{helper_command, helper_mode, HELPER_ENV};

#[cfg(test)]
mod tests;

/// Make `link` point at `target`; false where the platform refuses.
#[cfg(unix)]
pub fn link(target: &Path, link: &Path) -> bool {
    std::os::unix::fs::symlink(target, link).is_ok()
}

/// Make `link` point at `target`; false (with the reason printed) where Windows refuses because
/// the account lacks the symlink privilege.
#[cfg(windows)]
pub fn link(target: &Path, link: &Path) -> bool {
    let made = if target.is_dir() {
        std::os::windows::fs::symlink_dir(target, link)
    } else {
        std::os::windows::fs::symlink_file(target, link)
    };
    if let Err(e) = &made {
        eprintln!("SKIP: no symlinks here: {e}");
    }
    made.is_ok()
}

/// Hands out `data`, then fails every read with "disk gone".
pub struct DiesAfter(pub &'static [u8]);

impl Read for DiesAfter {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.0.is_empty() {
            return Err(io::Error::other("disk gone"));
        }
        let n = self.0.len().min(buf.len());
        buf[..n].copy_from_slice(&self.0[..n]);
        self.0 = &self.0[n..];
        Ok(n)
    }
}

/// An `open` that is refused, as a file without permission would be.
pub fn denied(_: &Path) -> io::Result<Box<dyn Read>> {
    Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
}

/// A writer that breaks, as a full disk does.
pub enum BrokenWriter {
    /// Every write fails.
    Write,
    /// Writes are taken; every flush fails.
    Flush,
}

impl io::Write for BrokenWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            BrokenWriter::Write => Err(io::Error::other("no space left")),
            BrokenWriter::Flush => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("no space left"))
    }
}
