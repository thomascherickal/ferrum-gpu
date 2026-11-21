use super::*;

#[test]
fn a_check_passes_only_below_its_tolerance_and_never_with_nan() {
    let c = |error| Check {
        name: "x".into(),
        error,
        tolerance: 1e-5,
    };
    assert!(c(0.0).ok() && c(9.9e-6).ok());
    assert!(!c(1e-5).ok() && !c(1.0).ok() && !c(f32::NAN).ok() && !c(f32::INFINITY).ok());
}

#[test]
fn a_report_passes_when_every_check_does_and_there_is_one() {
    assert!(report().passed());
    let mut r = report();
    r.checks[1].error = 1.0;
    assert!(!r.passed());
    r.checks.clear();
    assert!(!r.passed(), "no checks is not a pass");
}

#[test]
fn the_text_names_the_machine_the_checks_the_speeds_and_the_verdict() {
    let t = report().text();
    for want in [
        "lab report 2026-11-17 on linux x86_64 (wgpu 30.0.1)",
        "adapter: Fake GPU (Integrated, Metal)",
        "driver: fake-driver 1.2",
        "features: f16, subgroups; largest buffer 2047 MiB",
        "memory budget: 7000 MiB (an estimate)",
        "PASS f32: max relative error 2.00e-7 (allowed 1e-5)",
        "f32: model buffers 12.5 MiB, decode 123 tokens/s, prefill 4321 tokens/s",
        "q4_0: model buffers 3.1 MiB, decode 8.6 tokens/s, prefill 46 tokens/s",
        "CPU reference prefill, f32: 900 tokens/s",
        "note: random weights; speed only",
        "result: PASS",
    ] {
        assert!(t.contains(want), "missing {want:?} in\n{t}");
    }
    assert!(t.contains("median of 5 runs"));
}

#[test]
fn a_failing_check_and_missing_facts_are_said_plainly() {
    let mut r = report();
    r.checks[0].error = 3e-3;
    r.adapter = None;
    r.speeds.clear();
    r.cpu_prefill_tps = None;
    let t = r.text();
    assert!(t.contains("FAIL f32") && t.contains("result: FAIL"), "{t}");
    assert!(t.contains("adapter: none (no GPU was used)"));
    assert!(!t.contains("speed ("), "no speed section without speeds");
    // an adapter that reports no driver or features
    let mut r = report();
    let a = r.adapter.as_mut().unwrap();
    a.driver.clear();
    a.features.clear();
    let t = r.text();
    assert!(
        t.contains("driver: not reported") && t.contains("features: none reported"),
        "{t}"
    );
    // no CPU figure, no CPU line
    r.cpu_prefill_tps = None;
    assert!(!r.text().contains("CPU reference prefill"));
}

#[test]
fn rates_are_written_without_false_precision() {
    assert_eq!(rate(1234.4), "1234");
    assert_eq!(rate(100.0), "100");
    assert_eq!(rate(56.4), "56");
    assert_eq!(rate(7.84), "7.8");
    assert_eq!(rate(0.0), "0.0");
    assert_eq!(rate(f64::NAN), "n/a");
    assert_eq!(rate(f64::INFINITY), "n/a");
    assert_eq!(rate(-1.0), "n/a");
}
