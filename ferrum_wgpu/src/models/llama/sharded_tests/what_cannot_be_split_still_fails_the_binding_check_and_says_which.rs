use super::*;

#[test]
fn what_cannot_be_split_still_fails_the_binding_check_and_says_which() {
    let c = cfg();
    // below the KV cache (640 bytes) but above one column (128 bytes)
    let e = plan_limited(&c, None, 300)
        .check(300, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("cache") && e.contains("300"), "{e}");
}

#[test]
fn a_limit_below_one_column_is_refused_when_building_the_model() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 2, false, false);
    let e = GpuLlama::with_binding_limit(&g, &w, None, None, 100)
        .err()
        .expect("refused")
        .to_string();
    assert!(e.contains("binding limit") || e.contains("cache"), "{e}");
}

#[test]
fn the_default_constructor_uses_the_adapters_own_limit() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 2, false, false);
    let m = GpuLlama::new(&g, &w, None, None).unwrap();
    assert_eq!(m.binding_limit(), g.info().max_binding);
}

#[test]
fn a_token_outside_the_blocks_is_an_error_not_a_wrong_row() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 2, false, false);
    let m = GpuLlama::with_binding_limit(&g, &w, None, None, LIMIT).unwrap();
    let x = g.zeros(32).unwrap();
    let gi = g.zeros(1).unwrap();
    g.begin_batch().unwrap();
    let e = m
        .tok
        .gather(&g, (&[17], &gi), &x, 32)
        .expect_err("refused")
        .to_string();
    g.end_batch(false).unwrap();
    assert!(e.contains("outside the embedding"), "{e}");
}

#[test]
fn a_weight_stored_in_blocks_occupies_the_bytes_of_the_whole_weight() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 4, false, false);
    let whole = GpuLlama::new(&g, &w, None, None).unwrap();
    let cut = GpuLlama::with_binding_limit(&g, &w, None, None, LIMIT).unwrap();
    assert!(matches!(cut.blocks[0].wq, Lin::Sharded { .. }));
    assert_eq!(
        cut.blocks[0].wq.device_bytes(),
        whole.blocks[0].wq.device_bytes()
    );
}
