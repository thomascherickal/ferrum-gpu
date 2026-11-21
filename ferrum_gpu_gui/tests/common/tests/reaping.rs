use super::Group;
use crate::common::*;
use std::time::{Duration, Instant};

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
