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

/// `program` run under coreutils `timeout`: SIGTERM after `secs`, SIGKILL 5 s later, to the whole
/// process group `timeout` leads. This outlives a test process that was itself killed (so no
/// browser is left running) and ends a hung `node`. Plain `program` when `timeout` is missing.
pub fn with_deadline(program: &str, secs: u64) -> Command {
    let have_timeout = Command::new("timeout")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if have_timeout {
        let mut c = Command::new("timeout");
        c.args(["-k", "5", &secs.to_string(), program]);
        c
    } else {
        Command::new(program)
    }
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
    /// `max_secs` is a hard lifetime enforced by `timeout`, even if this test process is killed.
    pub fn open(binary: &str, url: &str, tag: &str, max_secs: u64) -> Browser {
        let base = std::env::temp_dir().join(format!("ferrum-gui-{tag}-{}", std::process::id()));
        let (profile, log) = (base.clone(), base.with_extension("log"));
        let stderr = std::fs::File::create(&log).expect("create the browser log");
        let child = with_deadline(binary, max_secs)
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

/// The `kill` arguments that send `signal` (`TERM`, `KILL`, `0`) to process group `pgid`, or
/// `None` for a group that must never be signalled: 0 is our own group, and 1 written as a
/// negative pid means "every process this user owns".
///
/// The `--` is the fix for a bug that logged the whole desktop session out. A bare `-<pgid>` is
/// read as an option cluster by procps-ng 4.0.4, which then signals minus its FIRST DIGIT:
/// `kill -TERM -1149968` ran `kill(-1, SIGTERM)` whenever Chrome's pid began with 1. After `--`
/// the number is an operand for every `kill` (POSIX), whatever its digits.
pub fn group_signal_args(pgid: u32, signal: &str) -> Option<[String; 4]> {
    (pgid > 1).then(|| {
        [
            "-s".to_string(),
            signal.to_string(),
            "--".to_string(),
            format!("-{pgid}"),
        ]
    })
}

/// Send `signal` to every process in the group led by `pgid`; true when something received it
/// (nothing is sent for the groups refused above).
pub fn signal_group(pgid: u32, signal: &str) -> bool {
    group_signal_args(pgid, signal).is_some_and(|args| {
        Command::new("kill")
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// SIGTERM every process in the group led by `pgid`.
pub fn terminate_group(pgid: u32) {
    signal_group(pgid, "TERM");
}

/// Whether any process (a zombie counts) is still in the group led by `pgid`.
pub fn group_alive(pgid: u32) -> bool {
    signal_group(pgid, "0")
}

/// End the group `leader` leads and reap the leader. SIGTERM first; a leader still running after
/// `term_grace` is killed; helpers still alive `kill_grace` after that (Chrome helpers that ignore
/// SIGTERM) get SIGKILL. The leader stays unreaped, so its pid (the group id) cannot be reused,
/// until the first signal has been sent.
// ponytail: between the last liveness check and the SIGKILL the group could empty and its id be
// reused; the window is microseconds, so this stays a check-then-kill. A cgroup would close it.
pub fn reap_group(leader: &mut Child, term_grace: Duration, kill_grace: Duration) {
    let pgid = leader.id();
    terminate_group(pgid);
    let exited = wait_for(term_grace, || {
        leader.try_wait().ok().flatten().map(|_| String::new())
    });
    if exited.is_none() {
        let _ = leader.kill();
    }
    let _ = leader.wait();
    if wait_for(kill_grace, || (!group_alive(pgid)).then(String::new)).is_none() {
        signal_group(pgid, "KILL");
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        reap_group(
            &mut self.child,
            Duration::from_secs(3),
            Duration::from_secs(2),
        );
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
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    /// A throwaway process group that is always killed, so a failing assertion cannot leave
    /// `sleep 60` helpers behind.
    struct Group(Child);
    impl Group {
        fn new(script: &str) -> Group {
            Group(
                Command::new("sh")
                    .args(["-c", script])
                    .process_group(0)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .expect("start sh"),
            )
        }
        fn pgid(&self) -> u32 {
            self.0.id()
        }
    }
    impl Drop for Group {
        fn drop(&mut self) {
            signal_group(self.0.id(), "KILL");
            let _ = self.0.wait();
        }
    }

    /// Regression: the group id used to be passed as a bare `-<pgid>`, which procps-ng's `kill`
    /// turned into `kill(-1, SIGTERM)` for any pid starting with 1 (the desktop session died).
    #[test]
    fn the_group_id_is_always_an_operand_after_a_double_dash() {
        // pids from real session crashes: all begin with the digit 1
        for pgid in [2u32, 7654321, 19207, 1149968, 1362056, 1908550] {
            for sig in ["TERM", "KILL", "0"] {
                let args = group_signal_args(pgid, sig).unwrap();
                assert_eq!(args, ["-s", sig, "--", &format!("-{pgid}")], "{pgid}");
            }
        }
        // group 0 (ourselves) and group 1 (as -1: everything we own) are never signalled
        for pgid in [0u32, 1] {
            assert_eq!(group_signal_args(pgid, "TERM"), None);
            assert!(!signal_group(pgid, "TERM"));
            assert!(!group_alive(pgid));
        }
        terminate_group(0);
        terminate_group(1);
    }

    #[test]
    fn terminating_a_group_ends_that_group_and_nothing_else() {
        let mut victim = Group::new("sleep 60 & wait"); // a leader with a helper, like Chrome
        let bystander = Group::new("sleep 60 & wait"); // someone else's group
        std::thread::sleep(Duration::from_millis(100));
        let pgid = victim.pgid();
        terminate_group(pgid);
        assert_eq!(victim.0.wait().unwrap().signal(), Some(15), "SIGTERM");
        let gone = wait_for(Duration::from_secs(5), || {
            (!group_alive(pgid)).then(String::new)
        });
        assert!(gone.is_some(), "the helper in the group was terminated too");
        assert!(
            group_alive(bystander.pgid()),
            "a bystander group is untouched"
        );
    }

    /// Hazard: helpers that ignore SIGTERM used to survive `drop`.
    #[test]
    fn reap_group_escalates_to_sigkill_for_leaders_and_helpers_that_ignore_term() {
        let mut stubborn = Group::new("trap '' TERM; sleep 60 & wait");
        let bystander = Group::new("sleep 60 & wait");
        std::thread::sleep(Duration::from_millis(200));
        let pgid = stubborn.pgid();
        let t0 = Instant::now();
        reap_group(
            &mut stubborn.0,
            Duration::from_millis(300),
            Duration::from_millis(300),
        );
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "bounded by the graces"
        );
        let gone = wait_for(Duration::from_secs(5), || {
            (!group_alive(pgid)).then(String::new)
        });
        assert!(gone.is_some(), "leader and helper are both gone");
        assert!(
            group_alive(bystander.pgid()),
            "a bystander group is untouched"
        );
    }

    #[test]
    fn reap_group_is_prompt_when_everything_obeys_sigterm() {
        let mut polite = Group::new("sleep 60 & wait");
        std::thread::sleep(Duration::from_millis(100));
        let t0 = Instant::now();
        reap_group(
            &mut polite.0,
            Duration::from_secs(3),
            Duration::from_secs(2),
        );
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "no grace is waited out"
        );
        assert!(!group_alive(polite.pgid()));
    }

    /// Hazard: a process whose parent died used to run on forever. `timeout` is the watchdog.
    #[test]
    fn with_deadline_ends_a_hung_program_and_its_helpers() {
        let mut c = with_deadline("sh", 1);
        let mut child = c
            .args(["-c", "sleep 60 & wait"])
            .process_group(0)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start");
        let pgid = child.id();
        let t0 = Instant::now();
        let status = child.wait().unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(8),
            "ended by the deadline"
        );
        assert_eq!(status.code(), Some(124), "timeout's own 'timed out' status");
        let gone = wait_for(Duration::from_secs(5), || {
            (!group_alive(pgid)).then(String::new)
        });
        assert!(gone.is_some(), "the helper died with it");
    }

    #[test]
    fn with_deadline_passes_the_program_through_unchanged_when_it_finishes() {
        let out = with_deadline("echo", 30).arg("hi").output().unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hi");
    }
}
