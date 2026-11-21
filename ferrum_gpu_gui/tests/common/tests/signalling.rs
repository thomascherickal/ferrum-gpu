use super::Group;
use crate::common::*;
use std::os::unix::process::ExitStatusExt;
use std::time::Duration;

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
