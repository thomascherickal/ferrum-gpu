use super::wait::wait_for;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// The `kill` arguments that send `signal` (`TERM`, `KILL`, `0`) to process group `pgid`, or
/// `None` for a group that must never be signalled: 0 is our own group, and 1 written as a
/// negative pid means "every process this user owns".
///
/// The `--` is the fix for a bug that logged the whole desktop session out. A bare `-<pgid>` is
/// read as an option cluster by procps-ng 4.0.4, which then signals minus its FIRST DIGIT:
/// `kill -TERM -1149968` ran `kill(-1, SIGTERM)` whenever Chrome's pid began with 1. After `--`
/// the number is an operand for every `kill` (POSIX), whatever its digits.
pub(crate) fn group_signal_args(pgid: u32, signal: &str) -> Option<[String; 4]> {
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
pub(crate) fn signal_group(pgid: u32, signal: &str) -> bool {
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
pub(crate) fn terminate_group(pgid: u32) {
    signal_group(pgid, "TERM");
}

/// Whether any process (a zombie counts) is still in the group led by `pgid`.
pub(crate) fn group_alive(pgid: u32) -> bool {
    signal_group(pgid, "0")
}

/// End the group `leader` leads and reap the leader. SIGTERM first; a leader still running after
/// `term_grace` is killed; helpers still alive `kill_grace` after that (Chrome helpers that ignore
/// SIGTERM) get SIGKILL. The leader stays unreaped, so its pid (the group id) cannot be reused,
/// until the first signal has been sent.
// ponytail: between the last liveness check and the SIGKILL the group could empty and its id be
// reused; the window is microseconds, so this stays a check-then-kill. A cgroup would close it.
pub(crate) fn reap_group(leader: &mut Child, term_grace: Duration, kill_grace: Duration) {
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
