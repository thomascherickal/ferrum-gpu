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
fn tie_with_an_explicit_gpu_is_refused_by_name() {
    let (kind, _) = run("--device gpu --tie");
    let e = kind.unwrap_err();
    assert!(e.contains("--tie") && e.contains("CPU engine only"), "{e}");
}

#[test]
fn dropout_with_an_explicit_gpu_is_refused_by_name() {
    let e = run("--device gpu --dropout 0.1").0.unwrap_err();
    assert!(
        e.contains("--dropout") && e.contains("CPU engine only"),
        "{e}"
    );
}

#[test]
fn both_options_are_named_together() {
    let e = run("--device gpu --tie --dropout 0.1").0.unwrap_err();
    assert!(e.contains("--tie and --dropout"), "{e}");
}

#[test]
fn auto_falls_back_to_the_cpu_and_says_so() {
    for flags in ["--tie", "--dropout 0.2", "--device auto --tie"] {
        let (kind, out) = run(flags);
        assert_eq!(kind.unwrap(), "cpu", "{flags}");
        assert!(out.contains("training on the CPU engine"), "{flags}: {out}");
    }
}

#[test]
fn a_rate_above_the_maximum_is_announced_as_clamped() {
    let (_, out) = run("--device cpu --dropout 0.99");
    assert!(out.contains("is used as 0.95"), "{out}");
    let (_, quiet) = run("--device cpu --dropout 0.5");
    assert!(!quiet.contains("is used as"), "{quiet}");
}
