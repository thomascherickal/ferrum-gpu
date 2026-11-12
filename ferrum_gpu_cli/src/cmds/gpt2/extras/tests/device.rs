use super::args;
use crate::cmds::gpt2::extras::{app_for, parse};

fn run(flags: &str) -> (Result<String, String>, String) {
    let a = args(flags);
    let x = parse(&a).unwrap();
    let mut out = Vec::new();
    let kind = app_for(&a, &x, &mut out).map(|app| app.backend.kind().to_string());
    (kind, String::from_utf8(out).unwrap())
}

#[test]
fn a_run_without_extras_keeps_the_ordinary_device_choice() {
    let (kind, out) = run("--device cpu");
    assert_eq!(kind.unwrap(), "cpu");
    assert!(out.is_empty(), "nothing is announced: {out}");
}

#[test]
fn the_extras_never_change_the_device_or_announce_a_fallback() {
    // both options train on either engine, so they are not a reason to leave the GPU
    for flags in [
        "--device cpu --tie",
        "--device cpu --dropout 0.2",
        "--device cpu --tie --dropout 0.2",
    ] {
        let (kind, out) = run(flags);
        assert_eq!(kind.unwrap(), "cpu", "{flags}");
        assert!(out.is_empty(), "{flags}: {out}");
    }
    for flags in [
        "--tie",
        "--dropout 0.2",
        "--device auto --tie --dropout 0.2",
    ] {
        let (_, out) = run(flags);
        assert!(!out.contains("CPU engine"), "{flags}: {out}");
    }
}

#[test]
fn a_rate_above_the_maximum_is_announced_as_clamped() {
    let (_, out) = run("--device cpu --dropout 0.99");
    assert!(out.contains("is used as 0.95"), "{out}");
    let (_, quiet) = run("--device cpu --dropout 0.5");
    assert!(!quiet.contains("is used as"), "{quiet}");
}
