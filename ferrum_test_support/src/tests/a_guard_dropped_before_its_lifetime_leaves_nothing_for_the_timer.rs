use super::*;
use std::process::Stdio;
use std::time::Duration;

#[test]
fn a_guard_dropped_before_its_lifetime_leaves_nothing_for_the_timer() {
    let mut cmd = helper_command("tests::guarded::helper_process", "sleep").unwrap();
    let guard = Guarded::spawn(
        &mut cmd,
        Stdio::null(),
        Stdio::null(),
        Some(Duration::from_millis(50)),
    )
    .unwrap();
    drop(guard);
    // the timer wakes after the guard is gone and finds nothing to end
    std::thread::sleep(Duration::from_millis(300));
}
