use super::*;

#[test]
fn an_allocation_failure_while_enabling_qat_is_reported_and_leaves_it_off() {
    let Some(g) = open_for_test() else { return };
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(cfg(), 55), None).unwrap();
    g.fail_allocs_after(1);
    let e = tr.set_qat(true).unwrap_err().to_string();
    g.fail_allocs_after(usize::MAX);
    assert!(e.contains("injected fault"), "{e}");
    assert!(!tr.qat_enabled());
    tr.set_qat(true).unwrap();
}
