use super::*;

#[test]
fn an_invalid_config_or_a_failed_allocation_is_refused_at_construction() {
    let Some(g) = open_for_test() else { return };
    let mut w = Gpt2::random(cfg(), 2);
    w.cfg.heads = 5;
    assert!(GpuGpt2::new(&g, &w, None).is_err());
    let w = Gpt2::random(cfg(), 2);
    // fail the 1st, 2nd, ... allocation (the last is the first compile of the matmul kernel)
    for k in 0.. {
        let Some(g) = open_for_test() else { return };
        g.fail_allocs_after(k);
        match GpuGpt2::new(&g, &w, None) {
            Err(e) => assert!(e.to_string().contains("injected fault"), "{k}: {e}"),
            Ok(_) => break,
        }
    }
}
