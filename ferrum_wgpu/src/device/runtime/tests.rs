use super::*;
use crate::budget::Source;
use crate::device::open_for_test;

#[test]
fn groups_cover_and_respect_the_limit() {
    assert_eq!(groups_for((17, 9), (8, 8), 65535).unwrap(), (3, 2));
    assert_eq!(groups_for((0, 0), (8, 8), 65535).unwrap(), (0, 0));
    assert_eq!(groups_for((8, 8), (0, 0), 65535).unwrap(), (8, 8)); // zero tile is clamped
    let e = groups_for((8, 8 * 70_000), (8, 8), 65535).unwrap_err();
    assert!(e.to_string().contains("65535"));
}

#[test]
fn fault_injection_fails_exactly_the_requested_command_and_can_be_turned_off() {
    let Some(g) = open_for_test() else { return };
    let (a, b) = (g.upload(&[1.0, 2.0]).unwrap(), g.zeros(2).unwrap());
    g.fail_after(2);
    g.copy_f32(&a, 0, &b, 0, 1).unwrap(); // command 1
    g.copy_f32(&a, 1, &b, 1, 1).unwrap(); // command 2
    let e = g.copy_f32(&a, 0, &b, 0, 2).unwrap_err(); // command 3 fails
    assert!(e.to_string().contains("injected fault"));
    assert!(g.copy_f32(&a, 0, &b, 0, 2).is_err(), "and every later one");
    g.fail_after(usize::MAX);
    g.copy_f32(&a, 0, &b, 0, 2).unwrap();
    assert_eq!(g.read(&b).unwrap(), vec![1.0, 2.0]);
}

#[test]
fn a_failure_inside_a_batch_can_be_discarded_and_the_device_keeps_working() {
    let Some(g) = open_for_test() else { return };
    let (a, b) = (g.upload(&[5.0]).unwrap(), g.zeros(1).unwrap());
    g.begin_batch().unwrap();
    g.copy_f32(&a, 0, &b, 0, 1).unwrap();
    g.fail_after(0);
    assert!(g.copy_f32(&a, 0, &b, 0, 1).is_err());
    g.end_batch(false).unwrap();
    g.fail_after(usize::MAX);
    assert!(!g.batching());
    assert_eq!(
        g.read(&b).unwrap(),
        vec![0.0],
        "the discarded batch never ran"
    );
}

#[test]
fn groups_1d_refuses_sizes_beyond_two_dimensions_of_full_rows() {
    let e = groups_1d(usize::MAX / 2).unwrap_err();
    assert!(e.to_string().contains("1-D dispatch limit"));
    let max = GROUPS_X * GROUPS_X * WG_1D;
    assert!(groups_1d(max).is_ok());
    assert!(groups_1d(max + WG_1D).is_err());
}

#[test]
fn groups_1d_spills_into_y_only_past_the_x_limit() {
    assert_eq!(groups_1d(0).unwrap(), (1, 1));
    assert_eq!(groups_1d(1).unwrap(), (1, 1));
    assert_eq!(groups_1d(64).unwrap(), (1, 1));
    assert_eq!(groups_1d(65).unwrap(), (2, 1));
    let edge = GROUPS_X * WG_1D; // 4,194,240: the last size that fits in one row
    assert_eq!(groups_1d(edge).unwrap(), (65535, 1));
    assert_eq!(
        groups_1d(edge + 1).unwrap(),
        (65535, 2),
        "spilling requires full rows: the shader stride is constant"
    );
    let (x, y) = groups_1d(10 * edge).unwrap();
    assert!(x as usize * y as usize * WG_1D >= 10 * edge);
    assert_eq!(x, 65535);
    // The shader's constant stride must equal one full row of invocations.
    assert!(PRELUDE.contains(&(GROUPS_X * WG_1D).to_string()));
}

#[test]
fn wgpu_errors_map_to_readable_messages() {
    let src = || -> wgpu::ErrorSource { "x".into() };
    assert!(map_error(wgpu::Error::OutOfMemory { source: src() })
        .to_string()
        .contains("out of memory"));
    let v = map_error(wgpu::Error::Validation {
        source: src(),
        description: "bad bind".into(),
    });
    assert!(v.to_string().contains("bad bind"));
    let i = map_error(wgpu::Error::Internal {
        source: src(),
        description: "oops".into(),
    });
    assert!(i.to_string().contains("oops"));
}

#[test]
fn pipelines_are_compiled_once_per_key() {
    let Some(g) = open_for_test() else { return };
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<f32>; @compute @workgroup_size(1) fn main() { o[0] = 1.0; }";
    let before = g.pipelines_built();
    g.pipeline("test-once", src).unwrap();
    g.pipeline("test-once", src).unwrap();
    assert_eq!(g.pipelines_built(), before + 1);
}

#[test]
fn a_broken_shader_is_an_error_not_a_panic() {
    let Some(g) = open_for_test() else { return };
    let e = g
        .pipeline("test-broken", "this is not wgsl")
        .expect_err("must fail");
    assert!(
        e.to_string().contains("validation") || e.to_string().contains("internal"),
        "{e}"
    );
}

#[test]
fn dispatch_runs_a_kernel_through_the_generic_path() {
    let Some(g) = open_for_test() else { return };
    let src = "struct U { v: u32, p0: u32, p1: u32, p2: u32 }
        @group(0) @binding(0) var<storage, read_write> o: array<f32>;
        @group(0) @binding(1) var<uniform> u: U;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) i: vec3<u32>) { o[i.x] = f32(u.v + i.x); }";
    let p = g.pipeline("test-dispatch", src).unwrap();
    let out = g.zeros(4).unwrap();
    g.dispatch(&p, &[&out], &[10], (4, 1)).unwrap();
    assert_eq!(g.read(&out).unwrap(), vec![10.0, 11.0, 12.0, 13.0]);
}

#[test]
fn a_plan_over_budget_allocates_nothing() {
    let Some(g) = open_for_test() else { return };
    let mut plan = BufferPlan::default();
    plan.add("a", 4096);
    plan.add("b", 4096);
    let before = g.buffers_created();
    let tight = Some(Budget {
        bytes: 5000,
        source: Source::Override,
    });
    let e = g.alloc_plan(&plan, tight).expect_err("must refuse");
    assert!(e.to_string().contains("8192") && e.to_string().contains("5000"));
    assert_eq!(
        g.buffers_created(),
        before,
        "no buffer may be created on refusal"
    );
}

#[test]
fn a_batch_runs_commands_in_order_in_one_submit() {
    let Some(g) = open_for_test() else { return };
    let (a, b, c) = (
        g.upload(&[1.0, 2.0, 3.0]).unwrap(),
        g.zeros(3).unwrap(),
        g.zeros(3).unwrap(),
    );
    assert!(!g.batching());
    g.begin_batch().unwrap();
    assert!(g.batching());
    // c depends on b, which depends on a: only in-order execution gives c == a.
    g.copy_f32(&a, 0, &b, 0, 3).unwrap();
    g.copy_f32(&b, 0, &c, 0, 3).unwrap();
    assert!(g.read(&c).unwrap_err().to_string().contains("batch"));
    g.end_batch(true).unwrap();
    assert!(!g.batching());
    assert_eq!(g.read(&c).unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn a_discarded_batch_does_nothing_and_misuse_is_an_error() {
    let Some(g) = open_for_test() else { return };
    let (a, b) = (g.upload(&[5.0]).unwrap(), g.zeros(1).unwrap());
    g.begin_batch().unwrap();
    g.copy_f32(&a, 0, &b, 0, 1).unwrap();
    assert!(g
        .begin_batch()
        .unwrap_err()
        .to_string()
        .contains("already open"));
    g.end_batch(false).unwrap();
    assert_eq!(
        g.read(&b).unwrap(),
        vec![0.0],
        "discarded work must not run"
    );
    assert!(g
        .end_batch(true)
        .unwrap_err()
        .to_string()
        .contains("no batch"));
    // A dispatch inside a batch goes through the same generic path.
    let src = "struct U { v: u32, p0: u32, p1: u32, p2: u32 }
        @group(0) @binding(0) var<storage, read_write> o: array<f32>;
        @group(0) @binding(1) var<uniform> u: U;
        @compute @workgroup_size(1) fn main() { o[0] = f32(u.v); }";
    let p = g.pipeline("test-batch-dispatch", src).unwrap();
    let out = g.zeros(1).unwrap();
    g.begin_batch().unwrap();
    g.dispatch(&p, &[&out], &[7], (1, 1)).unwrap();
    g.end_batch(true).unwrap();
    assert_eq!(g.read(&out).unwrap(), vec![7.0]);
}

#[test]
fn a_bad_batched_dispatch_fails_at_end_batch_and_does_not_brick_the_device() {
    let Some(g) = open_for_test() else { return };
    // This shader declares no uniform, but dispatch always binds one: the bind group is invalid.
    let bad = "@group(0) @binding(0) var<storage, read_write> o: array<f32>; @compute @workgroup_size(1) fn main() { o[0] = 1.0; }";
    let p = g.pipeline("test-batch-bad", bad).unwrap();
    let out = g.zeros(1).unwrap();
    g.begin_batch().unwrap();
    g.dispatch(&p, &[&out], &[1], (1, 1)).unwrap(); // recording does not validate
    let e = g.end_batch(true).unwrap_err();
    assert!(e.to_string().contains("GPU error"), "{e}");
    // The error was reported once; the device keeps working.
    g.check().unwrap();
    assert_eq!(g.read(&g.upload(&[3.0]).unwrap()).unwrap(), vec![3.0]);
    // Device loss, by contrast, is sticky.
    g.destroy();
    crate::device::wait_for_loss(&g);
    assert!(g.check().unwrap_err().to_string().contains("device lost"));
    assert!(
        g.check().unwrap_err().to_string().contains("device lost"),
        "still lost on the second check"
    );
}

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
