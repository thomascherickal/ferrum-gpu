use super::*;

#[test]
fn the_command_prints_text_or_json_for_an_available_adapter() {
    let Some(_) = ferrum_wgpu::device::open_for_test() else {
        return;
    };
    // the adapter the test harness would choose, named so the command opens the same one
    let name = std::env::var("FERRUM_GPU_ADAPTER").ok();
    let mut base = vec!["lab", "--parity-only", "--reps", "1"];
    if let Some(n) = name.as_deref() {
        base.extend(["--gpu-adapter", n]);
    }
    let mut out = Vec::new();
    run_command(&argv(&base), &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(
        text.contains("parity against the CPU reference") && text.contains("result: PASS"),
        "{text}"
    );
    let mut json = base.clone();
    json.push("--json");
    let mut out = Vec::new();
    run_command(&argv(&json), &mut out).unwrap();
    let j = crate::json::Json::parse(String::from_utf8(out).unwrap().trim()).unwrap();
    assert_eq!(j.get("passed"), Some(&crate::json::Json::Bool(true)));
    assert_eq!(
        j.get("format").and_then(crate::json::Json::as_f64),
        Some(1.0)
    );
}

#[test]
fn lab_is_a_device_command_the_studio_may_run() {
    assert!(crate::jobs::allow::COMMANDS.contains(&"lab"));
    assert!(crate::jobs::allow::uses_device("lab"));
    assert!(crate::commands::USAGE.contains("lab"));
}

#[test]
fn text_is_a_failures_message() {
    assert_eq!(text("boom"), "boom");
    assert_eq!(text(std::io::Error::other("closed")), "closed");
}

#[test]
fn a_finished_report_is_printed_and_a_failed_check_is_an_error() {
    // a passing report prints as text or as json
    let mut out = Vec::new();
    finish(Ok(checked(1e-7)), false, &mut out).unwrap();
    assert!(String::from_utf8(out).unwrap().contains("PASS"));
    let mut out = Vec::new();
    finish(Ok(checked(1e-7)), true, &mut out).unwrap();
    assert!(String::from_utf8(out)
        .unwrap()
        .trim_start()
        .starts_with('{'));
    // a failed check is printed and then reported
    let mut out = Vec::new();
    let e = finish(Ok(checked(1.0)), false, &mut out).unwrap_err();
    assert_eq!(e, "lab: a parity check failed");
    assert!(String::from_utf8(out).unwrap().contains("FAIL"));
    // no report at all: the reason, and nothing printed
    let mut out = Vec::new();
    assert_eq!(
        finish(Err("no device".into()), false, &mut out).unwrap_err(),
        "no device"
    );
    assert!(out.is_empty());
    // a closed output is an error too
    struct Closed;
    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("closed"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    assert_eq!(
        finish(Ok(checked(1e-7)), false, &mut Closed).unwrap_err(),
        "closed"
    );
}
