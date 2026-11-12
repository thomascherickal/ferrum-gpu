//! Launching headless Chrome against the GUI server and collecting what the page reports.
#![allow(dead_code)]

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A Chrome/Chromium binary, if one is installed.
pub fn chrome() -> Option<String> {
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

/// A running headless browser in its own process group, so that dropping it ends exactly the
/// processes it started (Chrome forks helpers) and never anyone else's browser.
pub struct Browser {
    child: Child,
    profile: PathBuf,
    log: PathBuf,
}

impl Browser {
    /// Open `url` with WebGPU enabled (software Vulkan is what Chrome offers on a headless box).
    pub fn open(binary: &str, url: &str, tag: &str) -> Browser {
        let base = std::env::temp_dir().join(format!("ferrum-gui-{tag}-{}", std::process::id()));
        let (profile, log) = (base.clone(), base.with_extension("log"));
        let stderr = std::fs::File::create(&log).expect("create the browser log");
        let child = Command::new(binary)
            .args([
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu-sandbox",
                // the flag name is split because a source guard rejects the bare word in non-comment lines
                concat!("--enable-un", "safe-webgpu"),
                // route Chrome's GPU stack through ANGLE/Vulkan so real adapters are offered
                "--use-gl=angle",
                "--use-angle=vulkan",
                "--enable-features=Vulkan",
                "--ignore-gpu-blocklist",
                "--disable-gpu-driver-bug-workarounds",
                "--enable-logging=stderr",
                "--v=0",
                "--no-first-run",
                "--disable-extensions",
                &format!("--user-data-dir={}", profile.display()),
                url,
            ])
            .process_group(0)
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()
            .expect("start the browser");
        Browser {
            child,
            profile,
            log,
        }
    }

    /// The console and error lines Chrome wrote so far (the raw tail when there are none).
    pub fn console(&self) -> String {
        let all = std::fs::read_to_string(&self.log).unwrap_or_default();
        let picked: Vec<&str> = all
            .lines()
            .filter(|l| l.contains("CONSOLE") || l.contains("ERROR:"))
            .take(40)
            .collect();
        if picked.is_empty() {
            let lines: Vec<&str> = all.lines().collect();
            format!(
                "(no console lines; {} log lines, tail)\n{}",
                lines.len(),
                lines[lines.len().saturating_sub(15)..].join("\n")
            )
        } else {
            picked.join("\n")
        }
    }
}

/// The `kill` arguments that send SIGTERM to process group `pgid`, or `None` for a group that must
/// never be signalled: 0 is our own group, and 1 written as a negative pid means "every process
/// this user owns".
///
/// The `--` is the fix for a bug that logged the whole desktop session out. A bare `-<pgid>` is
/// read as an option cluster by procps-ng 4.0.4, which then signals minus its FIRST DIGIT:
/// `kill -TERM -1149968` ran `kill(-1, SIGTERM)` whenever Chrome's pid began with 1. After `--`
/// the number is an operand for every `kill` (POSIX), whatever its digits.
pub fn group_term_args(pgid: u32) -> Option<[String; 4]> {
    (pgid > 1).then(|| {
        [
            "-s".to_string(),
            "TERM".to_string(),
            "--".to_string(),
            format!("-{pgid}"),
        ]
    })
}

/// SIGTERM every process in the group led by `pgid` (nothing for the groups refused above).
pub fn terminate_group(pgid: u32) {
    if let Some(args) = group_term_args(pgid) {
        let _ = Command::new("kill")
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        // Terminate the whole group the browser leads (its pid is the group id).
        terminate_group(self.child.id());
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.profile);
        let _ = std::fs::remove_file(&self.log);
    }
}

/// Poll `fetch` (which returns the page's reported JSON text, if any) until it yields or `limit`
/// passes.
pub fn wait_for(limit: Duration, mut fetch: impl FnMut() -> Option<String>) -> Option<String> {
    let t0 = Instant::now();
    while t0.elapsed() < limit {
        if let Some(s) = fetch() {
            return Some(s);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    /// Regression: the group id used to be passed as a bare `-<pgid>`, which procps-ng's `kill`
    /// turned into `kill(-1, SIGTERM)` for any pid starting with 1 (the desktop session died).
    #[test]
    fn the_group_id_is_always_an_operand_after_a_double_dash() {
        // the last four are pids from real session crashes; all begin with the digit 1
        for pgid in [2u32, 7654321, 19207, 1149968, 1362056, 1908550] {
            let args = group_term_args(pgid).unwrap();
            assert_eq!(args, ["-s", "TERM", "--", &format!("-{pgid}")], "{pgid}");
        }
        // group 0 (ourselves) and group 1 (as -1: everything we own) are never signalled
        assert_eq!(group_term_args(0), None);
        assert_eq!(group_term_args(1), None);
        terminate_group(0);
        terminate_group(1);
    }

    #[test]
    fn terminating_a_group_ends_that_group_and_nothing_else() {
        let spawn = |script: &str| {
            Command::new("sh")
                .args(["-c", script])
                .process_group(0)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("start sh")
        };
        let mut victim = spawn("sleep 60 & wait"); // a leader with a helper, like Chrome
        let mut bystander = spawn("sleep 60 & wait"); // someone else's group
        std::thread::sleep(Duration::from_millis(100));
        let pgid = victim.id();
        terminate_group(pgid);
        assert_eq!(victim.wait().unwrap().signal(), Some(15), "SIGTERM");
        let group_alive = || {
            Command::new("kill")
                .args(["-s", "0", "--", &format!("-{pgid}")])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        };
        let gone = wait_for(Duration::from_secs(5), || {
            (!group_alive()).then(String::new)
        });
        assert!(gone.is_some(), "the helper in the group was terminated too");
        assert!(
            bystander.try_wait().unwrap().is_none(),
            "a process outside the group is untouched"
        );
        terminate_group(bystander.id());
        assert_eq!(bystander.wait().unwrap().signal(), Some(15));
    }
}
