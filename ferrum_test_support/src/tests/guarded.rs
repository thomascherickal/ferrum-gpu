use super::*;
use std::process::Stdio;
use std::time::{Duration, Instant};

const HELPER: &str = "tests::guarded::helper_process";

/// Not a test of its own: the part a re-executed test binary plays, chosen by [`HELPER_ENV`].
#[test]
fn helper_process() {
    let nap = |ms| std::thread::sleep(Duration::from_millis(ms));
    match helper_mode().as_deref() {
        Some("echo") => println!("hi from the helper"),
        Some("sleep") => nap(20_000),
        Some("tree") => {
            let mut grandchild = helper_command(HELPER, "sleep").unwrap().spawn().unwrap();
            grandchild.wait().unwrap();
        }
        Some("heartbeat") => loop {
            let path = std::env::var("FERRUM_TEST_BEATS").unwrap();
            let mut beats = std::fs::OpenOptions::new();
            let mut file = beats.append(true).create(true).open(path).unwrap();
            std::io::Write::write_all(&mut file, b".").unwrap();
            nap(20);
        },
        _ => {}
    }
}

#[test]
fn a_program_that_finishes_in_time_gives_its_output_unchanged() {
    let cmd = helper_command(HELPER, "echo").unwrap();
    let out = Deadlined::new(cmd, 30).output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("hi from the helper"));
    let missing = with_deadline("ferrum-no-such-program", 5)
        .arg("x")
        .args(["y"])
        .output();
    assert!(missing.is_err());
}

#[test]
fn the_deadline_ends_a_hung_program_and_its_helpers() {
    let cmd = helper_command(HELPER, "tree").unwrap();
    let started = Instant::now();
    let out = Deadlined::new(cmd, 1)
        .stdin(Stdio::null())
        .env("A", "b")
        .output()
        .unwrap();
    assert!(!out.status.success(), "killed, not finished");
    // the output pipe closes only when the helper died too; it would sleep for 20 s
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "ended by the deadline"
    );
}

#[test]
fn a_guarded_child_ends_when_its_lifetime_passes() {
    let mut cmd = helper_command(HELPER, "sleep").unwrap();
    let guard = Guarded::spawn(
        &mut cmd,
        Stdio::null(),
        Stdio::null(),
        Some(Duration::from_millis(300)),
    );
    let guard = guard.unwrap();
    assert!(!guard.exited());
    let started = Instant::now();
    while !guard.exited() && started.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(guard.exited(), "the watchdog killed it");
}

#[test]
fn dropping_a_guarded_child_stops_it_for_good() {
    let beats = std::env::temp_dir().join(format!("ferrum_beats_{}", std::process::id()));
    let mut cmd = helper_command(HELPER, "heartbeat").unwrap();
    cmd.env("FERRUM_TEST_BEATS", &beats);
    let guard = Guarded::spawn(&mut cmd, Stdio::null(), Stdio::null(), None).unwrap();
    while std::fs::metadata(&beats).map_or(0, |m| m.len()) < 3 {
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(guard);
    let size = || std::fs::metadata(&beats).map_or(0, |m| m.len());
    let at_drop = size();
    std::thread::sleep(Duration::from_millis(400));
    assert!(size() <= at_drop + 1, "no beat after the drop");
}

#[test]
fn a_background_command_runs_until_it_is_dropped() {
    let mut cmd = Deadlined::new(helper_command(HELPER, "sleep").unwrap(), 60);
    let guard = cmd.guard(Stdio::null(), Stdio::null()).unwrap();
    assert!(!guard.exited());
}
