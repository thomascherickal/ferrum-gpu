use super::*;

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
