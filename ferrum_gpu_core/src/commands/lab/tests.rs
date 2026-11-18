use super::*;
use crate::commands::run as run_command;

fn argv(words: &[&str]) -> Vec<String> {
    words.iter().map(|s| s.to_string()).collect()
}

struct NoReaders;
impl sysmem::Readers for NoReaders {
    fn nvidia_smi(&self) -> Option<String> {
        None
    }
    fn meminfo(&self) -> Option<String> {
        None
    }
    fn sysctl_memsize(&self) -> Option<String> {
        None
    }
    fn sysfs_vram_mb(&self) -> Vec<u64> {
        vec![]
    }
}

#[test]
fn bad_flags_are_refused_before_any_device_is_opened() {
    let mut out = Vec::new();
    for (args, needle) in [
        (
            vec!["lab", "--reps", "0"],
            "--reps must be between 1 and 200",
        ),
        (
            vec!["lab", "--reps", "201"],
            "--reps must be between 1 and 200",
        ),
        (vec!["lab", "--reps", "many"], "--reps"),
        (vec!["lab", "--size", "huge"], "small, medium"),
        (vec!["lab", "--nope"], "--nope"),
    ] {
        let e = run_command(&argv(&args), &mut out).unwrap_err();
        assert!(e.contains(needle), "{args:?}: {e}");
    }
    assert!(out.is_empty());
}

#[test]
fn no_adapter_is_an_error_that_says_what_lab_needs() {
    let e = run_command(
        &argv(&["lab", "--gpu-adapter", "no-such-adapter-anywhere"]),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(e.contains("lab needs a GPU adapter"), "{e}");
}

#[test]
fn the_text_report_on_an_available_adapter_passes_and_names_everything() {
    let Some(gpu) = ferrum_wgpu::device::open_for_test() else {
        return;
    };
    let r = build(
        &gpu,
        (speed::Size::Small, 1, true),
        &NoReaders,
        1_794_873_600,
    )
    .unwrap();
    assert!(r.passed(), "{}", r.text());
    assert_eq!(r.when, "2026-11-17");
    assert_eq!(r.checks.len(), 10);
    assert_eq!(r.speeds.len(), 8);
    assert!(r.cpu_prefill_tps.unwrap() > 0.0);
    assert_eq!(r.adapter.as_ref().unwrap().name, gpu.info().name);
    assert!(r.adapter.as_ref().unwrap().budget.contains("memory"));
    assert!(r.notes[0].contains("launch and dispatch"));
    assert!(r.text().contains("result: PASS"));
}

#[test]
fn parity_only_skips_the_timings() {
    let Some(gpu) = ferrum_wgpu::device::open_for_test() else {
        return;
    };
    let r = build(&gpu, (speed::Size::Small, 1, false), &NoReaders, 0).unwrap();
    assert!(r.speeds.is_empty() && r.cpu_prefill_tps.is_none() && r.notes.is_empty());
    assert!(r.passed());
}

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
