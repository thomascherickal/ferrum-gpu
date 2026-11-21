use crate::common::*;
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::time::{Duration, Instant};

/// Hazard: a process whose parent died used to run on forever. `timeout` is the watchdog.
#[test]
fn with_deadline_ends_a_hung_program_and_its_helpers() {
    if !has_timeout() {
        eprintln!(
            "SKIP: no coreutils `timeout` here (macOS has none unless coreutils is installed)"
        );
        return;
    }
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
