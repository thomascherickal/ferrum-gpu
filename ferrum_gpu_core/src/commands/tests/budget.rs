use super::fixtures::*;
use super::*;
use ferrum_wgpu::budget::BufferPlan;

#[test]
fn a_gpu_run_that_fits_prints_the_estimate_and_one_that_does_not_is_refused() {
    let m = fresh_model("tiny", 1).unwrap();
    let plan = preflight::train_plan(&m, 2, 16);
    let gpu = session_on(Arc::new(FakeGpu), listed());
    let mut out = Vec::new();
    gate(&mut out, &gpu, &plan, Some(4096)).unwrap();
    assert!(String::from_utf8(out)
        .unwrap()
        .contains("preflight: GPU buffers"));
    let e = gate(&mut Vec::new(), &gpu, &plan, Some(0)).unwrap_err();
    assert!(e.contains("refusing to start on Fake GPU"), "{e}");
}

#[test]
fn the_preflight_uses_the_cards_free_memory_with_the_measured_headroom() {
    // capability.md: on this card plans up to 1449 MiB loaded and plans from 1721 MiB failed
    let gpu = session_on(Arc::new(FakeGpu), listed());
    for fits in [279u64, 956, 1449] {
        let mut out = Vec::new();
        gate_with(&Mx330, &mut out, &gpu, &plan_of(fits), None)
            .unwrap_or_else(|e| panic!("{fits} MiB should pass: {e}"));
        assert!(
            String::from_utf8(out).unwrap().contains("budget 1591"),
            "the budget is shown"
        );
    }
    for refused in [1721u64, 1925, 2454, 4286] {
        let e = gate_with(&Mx330, &mut Vec::new(), &gpu, &plan_of(refused), None)
            .expect_err(&format!("{refused} MiB should be refused"));
        assert!(e.contains("refusing to start on Fake GPU"), "{e}");
    }
    // the flag still overrides what the machine says
    gate_with(&Mx330, &mut Vec::new(), &gpu, &plan_of(1925), Some(4096)).unwrap();
}

#[test]
fn with_no_memory_figure_only_the_binding_limit_applies_as_before() {
    let gpu = session_on(Arc::new(FakeGpu), listed());
    let mut out = Vec::new();
    // 4286 MiB in buffers that each respect the binding limit: only the missing budget is at issue
    let mut big = BufferPlan::default();
    for i in 0..5 {
        big.add(&format!("part {i}"), 857 * 1048576);
    }
    gate_with(&Blind, &mut out, &gpu, &big, None).unwrap();
    assert!(String::from_utf8(out).unwrap().contains("budget unknown"));
}
