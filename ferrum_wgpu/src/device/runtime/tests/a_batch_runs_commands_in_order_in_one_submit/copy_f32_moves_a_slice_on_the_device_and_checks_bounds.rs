use super::*;

#[test]
fn copy_f32_moves_a_slice_on_the_device_and_checks_bounds() {
    let Some(g) = open_for_test() else { return };
    let src = g.upload(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let dst = g.upload(&[9.0; 6]).unwrap();
    g.copy_f32(&src, 1, &dst, 2, 3).unwrap();
    assert_eq!(g.read(&dst).unwrap(), vec![9.0, 9.0, 2.0, 3.0, 4.0, 9.0]);
    g.copy_f32(&src, 0, &dst, 0, 0).unwrap();
    assert!(g
        .copy_f32(&src, 3, &dst, 0, 3)
        .unwrap_err()
        .to_string()
        .contains("exceeds"));
    assert!(g.copy_f32(&src, 0, &dst, 5, 2).is_err());
}

#[test]
fn a_plan_that_fits_allocates_one_buffer_per_row() {
    let Some(g) = open_for_test() else { return };
    let mut plan = BufferPlan::default();
    plan.add("a", 16);
    plan.add("b", 6); // rounds up to 2 f32
    let bufs = g.alloc_plan(&plan, None).unwrap();
    assert_eq!(bufs.iter().map(|b| b.len()).collect::<Vec<_>>(), vec![4, 2]);
}

#[test]
fn a_destroyed_device_reports_lost_instead_of_panicking() {
    let Some(g) = open_for_test() else { return };
    g.check().unwrap();
    g.destroy();
    // The lost callback fires asynchronously; poll until it lands (bounded).
    crate::device::wait_for_loss(&g);
    assert!(g.check().is_err(), "device loss must be observed");
    let e = g.zeros(4).expect_err("use after loss must fail");
    assert!(e.to_string().contains("lost"), "{e}");
}

#[test]
fn a_huge_allocation_is_an_error_or_succeeds_but_never_panics() {
    let Some(g) = open_for_test() else { return };
    let n = (g.info().max_binding / 4) as usize;
    let r = g.zeros(n);
    eprintln!("huge allocation: {:?}", r.as_ref().map(|b| b.len()));
    assert!(r.as_ref().map_or(true, |b| b.len() == n));
    g.check().unwrap();
}

#[test]
fn dispatch_1d_refuses_a_size_beyond_the_1d_limit_without_recording() {
    let Some(g) = open_for_test() else { return };
    let p = g.pipeline("test-d1d", "struct U { a: u32, b: u32, c: u32, d: u32 }\n@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n@group(0) @binding(1) var<uniform> u: U;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) { o[0] = 1.0 + f32(u.a); }").unwrap();
    let out = g.zeros(1).unwrap();
    let e = g.dispatch_1d(&p, &[&out], &[0], usize::MAX).unwrap_err();
    assert!(e.to_string().contains("1-D dispatch limit"), "{e}");
    g.dispatch_1d(&p, &[&out], &[0], 1).unwrap();
    assert_eq!(g.read(&out).unwrap(), vec![1.0]);
}

#[test]
#[should_panic(expected = "never succeeded")]
fn sweep_faults_panics_when_the_operation_never_succeeds() {
    let Some(g) = open_for_test() else {
        panic!("never succeeded (no adapter: stand-in)")
    };
    g.sweep_faults(|| Err(GpuError("injected fault".into())));
}
