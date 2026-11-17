use super::*;

fn adapter() -> AdapterFacts {
    AdapterFacts {
        name: "Fake GPU".into(),
        backend: "Metal".into(),
        kind: "Integrated".into(),
        driver: "fake-driver 1.2".into(),
        max_binding_mib: 2047,
        features: vec!["f16".into(), "subgroups".into()],
        budget: "memory budget: 7000 MiB (an estimate)".into(),
    }
}

fn report() -> Report {
    Report {
        when: "2026-11-17".into(),
        os: "linux x86_64".into(),
        wgpu: "30.0.1".into(),
        adapter: Some(adapter()),
        model: "random 4-layer model".into(),
        reps: 5,
        checks: vec![
            Check {
                name: "f32".into(),
                error: 2.0e-7,
                tolerance: 1e-5,
            },
            Check {
                name: "q8_0".into(),
                error: 4.0e-7,
                tolerance: 1e-5,
            },
        ],
        speeds: vec![
            Speed {
                precision: "f32".into(),
                model_mib: 12.5,
                decode_tps: 123.4,
                prefill_tps: 4321.0,
            },
            Speed {
                precision: "q4_0".into(),
                model_mib: 3.1,
                decode_tps: 8.55,
                prefill_tps: 45.6,
            },
        ],
        cpu_prefill_tps: Some(900.0),
        notes: vec!["random weights; speed only".into()],
    }
}

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

#[test]
fn the_json_has_every_field_and_parses_back() {
    let j = report().json();
    let back = Json::parse(&j.to_string()).unwrap();
    assert_eq!(back.get("format").and_then(Json::as_f64), Some(1.0));
    assert_eq!(back.get("passed"), Some(&Json::Bool(true)));
    assert_eq!(back.get("wgpu").and_then(Json::as_str), Some("30.0.1"));
    let a = back.get("adapter").unwrap();
    assert_eq!(
        a.get("driver").and_then(Json::as_str),
        Some("fake-driver 1.2")
    );
    assert_eq!(
        a.get("max_binding_mib").and_then(Json::as_f64),
        Some(2047.0)
    );
    let Some(Json::Arr(checks)) = back.get("checks") else {
        panic!("{back}")
    };
    assert_eq!(checks.len(), 2);
    assert_eq!(checks[0].get("ok"), Some(&Json::Bool(true)));
    let Some(Json::Arr(speeds)) = back.get("speeds") else {
        panic!()
    };
    assert_eq!(
        speeds[1].get("precision").and_then(Json::as_str),
        Some("q4_0")
    );
    assert_eq!(
        back.get("cpu_prefill_tps").and_then(Json::as_f64),
        Some(900.0)
    );
    // without an adapter or a CPU figure those are null, and a failing check says so
    let mut r = report();
    r.adapter = None;
    r.cpu_prefill_tps = None;
    r.checks[0].error = 1.0;
    let j = r.json();
    assert_eq!(j.get("adapter"), Some(&Json::Null));
    assert_eq!(j.get("cpu_prefill_tps"), Some(&Json::Null));
    assert_eq!(j.get("passed"), Some(&Json::Bool(false)));
}

#[test]
fn dates_are_right_at_known_points() {
    assert_eq!(date_of(0), "1970-01-01");
    assert_eq!(date_of(86399), "1970-01-01");
    assert_eq!(date_of(86400), "1970-01-02");
    assert_eq!(date_of(951_782_400), "2000-02-29"); // a leap day
    assert_eq!(date_of(1_709_164_800), "2024-02-29");
    assert_eq!(date_of(1_709_251_200), "2024-03-01");
    assert_eq!(date_of(4_102_444_800), "2100-01-01"); // 2100 is not a leap year
    assert_eq!(date_of(1_794_873_600), "2026-11-17");
}

#[test]
fn a_version_is_read_from_the_lock_text() {
    let lock = "[[package]]\nname = \"wgpu\"\nversion = \"30.0.1\"\n\n[[package]]\nname = \"wgpu-core\"\nversion = \"29.0.0\"\n";
    assert_eq!(locked_version(lock, "wgpu"), "30.0.1");
    assert_eq!(locked_version(lock, "wgpu-core"), "29.0.0");
    assert_eq!(locked_version(lock, "missing"), "unknown");
    assert_eq!(
        locked_version("name = \"wgpu\"\n", "wgpu"),
        "unknown",
        "a name with no version line"
    );
    assert_eq!(locked_version("", "wgpu"), "unknown");
}

#[test]
fn the_real_lock_file_names_the_wgpu_version_the_workspace_uses() {
    let v = locked_version(include_str!("../../../../../Cargo.lock"), "wgpu");
    assert_ne!(v, "unknown");
    assert!(v.chars().next().unwrap().is_ascii_digit(), "{v}");
}
