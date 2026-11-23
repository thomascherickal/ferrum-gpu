use std::process::{Command, Stdio};

/// A Chrome/Chromium binary, if one is installed.
pub(crate) fn chrome() -> Option<String> {
    [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]
    .into_iter()
    .find(|b| {
        Command::new(b)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
    .map(String::from)
}

/// Is GNU coreutils `timeout` installed? (Linux: yes; macOS: only with Homebrew coreutils.)
pub(crate) fn has_timeout() -> bool {
    Command::new("timeout")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// `program` run under coreutils `timeout`: SIGTERM after `secs`, SIGKILL 5 s later, to the whole
/// process group `timeout` leads. This outlives a test process that was itself killed (so no
/// browser is left running) and ends a hung `node`. Plain `program` when `timeout` is missing.
pub(crate) fn with_deadline(program: &str, secs: u64) -> Command {
    if has_timeout() {
        let mut c = Command::new("timeout");
        c.args(["-k", "5", &secs.to_string(), program]);
        c
    } else {
        Command::new(program)
    }
}
