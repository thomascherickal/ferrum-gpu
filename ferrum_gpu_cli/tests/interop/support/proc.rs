//! Running tools: under a deadline, and servers in their own process group that are ended without
//! ever signalling anything else.
//!
//! The group signals follow CLAUDE.md rule 16 (`kill -s TERM -- -<pgid>`, never a bare negative
//! number, never group 0 or 1). `ferrum_gpu_gui/tests/common` has the same logic for Chrome; this
//! is a second copy because the two live in different test crates.

use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// `program` under coreutils `timeout` (SIGTERM after `secs`, SIGKILL 5 s later), or plain when
/// `timeout` is not installed.
pub fn with_deadline(program: &Path, secs: u64) -> Command {
    let have = Command::new("timeout")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if have {
        let mut c = Command::new("timeout");
        c.args(["-k", "5", &secs.to_string()]).arg(program);
        c
    } else {
        Command::new(program)
    }
}

fn signal_group(pgid: u32, sig: &str) -> bool {
    pgid > 1
        && Command::new("kill")
            .args(["-s", sig, "--", &format!("-{pgid}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
}

/// A child in its own process group, ended with its whole group when dropped.
pub struct Group(pub Child);

impl Group {
    /// Start `cmd` as the leader of a new process group.
    pub fn spawn(cmd: &mut Command) -> Group {
        Group(
            cmd.process_group(0)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("start the tool"),
        )
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        let pgid = self.0.id();
        signal_group(pgid, "TERM");
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(3) && self.0.try_wait().ok().flatten().is_none() {
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
        // helpers that outlive their leader (a model runner) are killed too
        let t0 = Instant::now();
        while signal_group(pgid, "0") && t0.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(50));
        }
        if signal_group(pgid, "0") {
            signal_group(pgid, "KILL");
        }
    }
}

/// Poll `ready` every 100 ms until it is true or `secs` pass.
pub fn wait_until(secs: u64, mut ready: impl FnMut() -> bool) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(secs) {
        if ready() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}
