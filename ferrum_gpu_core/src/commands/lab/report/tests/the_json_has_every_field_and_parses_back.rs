use super::*;

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
    let v = locked_version(include_str!("../../../../../../Cargo.lock"), "wgpu");
    assert_ne!(v, "unknown");
    assert!(v.chars().next().unwrap().is_ascii_digit(), "{v}");
}
