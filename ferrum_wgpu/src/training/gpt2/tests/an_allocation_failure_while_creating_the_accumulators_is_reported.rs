use super::*;

#[test]
fn an_allocation_failure_while_creating_the_accumulators_is_reported() {
    let Some(g) = open_for_test() else { return };
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(cfg(), 54), None).unwrap();
    g.fail_allocs_after(1);
    let e = tr.ensure_accum().unwrap_err().to_string();
    g.fail_allocs_after(usize::MAX);
    assert!(e.contains("injected fault"), "{e}");
    tr.ensure_accum().unwrap();
}
