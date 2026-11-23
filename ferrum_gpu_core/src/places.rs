//! Where the process runs and where its user's files live.
use std::ffi::OsString;
use std::path::PathBuf;

/// The home directory according to `lookup` (`HOME`, else `USERPROFILE` as on Windows).
pub fn home_from(lookup: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    lookup("HOME")
        .or_else(|| lookup("USERPROFILE"))
        .map(PathBuf::from)
}

/// The home directory of this process's environment.
pub fn home_dir() -> Option<PathBuf> {
    home_from(|k| std::env::var_os(k))
}

/// The working directory, or `.` when it cannot be read (it was removed under the process).
pub fn cwd_or_dot() -> PathBuf {
    or_dot(std::env::current_dir())
}

fn or_dot(cwd: std::io::Result<PathBuf>) -> PathBuf {
    cwd.unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests;
