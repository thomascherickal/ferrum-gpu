use super::*;

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
