use super::*;

#[test]
fn an_invalid_config_or_a_failed_allocation_is_refused_at_construction() {
    let Some(g) = open_for_test() else { return };
    let mut w = Llama::random(cfg(), 2, true, false);
    w.cfg.heads = 0;
    assert!(GpuLlama::new(&g, &w, None, None).is_err());
    let w = Llama::random(cfg(), 2, true, false);
    // fail the 1st, 2nd, ... allocation (biases included; the last is the matmul kernel compile)
    for k in 0.. {
        let Some(g) = open_for_test() else { return };
        g.fail_allocs_after(k);
        match GpuLlama::new(&g, &w, None, None) {
            Err(e) => assert!(e.to_string().contains("injected fault"), "{k}: {e}"),
            Ok(_) => break,
        }
    }
}
