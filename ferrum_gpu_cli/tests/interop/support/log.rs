//! The step log of the real-tool tests: every step goes to stderr and, when `FERRUM_INTEROP_LOG`
//! names a file, is appended to it, so a run leaves a record of exactly what was done.

use std::io::Write;
use std::process::Command;

/// `[2026-11-14T19:40:07+05:30] message`, to stderr and to `$FERRUM_INTEROP_LOG`.
pub fn step(msg: impl AsRef<str>) {
    let now = Command::new("date")
        .arg("-Is")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let line = format!("[{now}] {}", msg.as_ref());
    eprintln!("{line}");
    if let Some(path) = std::env::var_os("FERRUM_INTEROP_LOG") {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Is the opt-in switch `var` set to `1`? When it is not, say how to turn the test on (and why it
/// is off by default) and return false.
pub fn opted_in(var: &str, what: &str) -> bool {
    let on = std::env::var(var).is_ok_and(|v| v == "1");
    if !on {
        eprintln!("SKIP: {what}; set {var}=1 to run it");
    }
    on
}

/// The lines of `text` that mention any of `needles`, trimmed, for the log.
pub fn grep(text: &str, needles: &[&str]) -> Vec<String> {
    text.lines()
        .filter(|l| needles.iter().any(|n| l.contains(n)))
        .map(|l| l.trim().chars().take(200).collect())
        .collect()
}
