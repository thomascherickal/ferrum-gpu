use super::*;

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
