use super::*;

#[test]
fn sharded_f32_decode_equals_the_whole_model_with_and_without_biases() {
    let Some(g) = open_for_test() else { return };
    for (biases, tied) in [(false, false), (true, true)] {
        let w = Llama::random(cfg(), 2, biases, tied);
        let whole = &mut GpuLlama::new(&g, &w, None, None).unwrap();
        let cut = &mut GpuLlama::with_binding_limit(&g, &w, None, None, LIMIT).unwrap();
        let (tok, head, gate) = blocks_of(cut);
        assert!(tok > 1 && head > 1 && gate > 1, "{tok} {head} {gate}");
        assert_eq!(blocks_of(whole), (1, 1, 1));
        let want = w.forward_logits(&IDS).unwrap();
        for (i, &t) in IDS.iter().enumerate() {
            let (a, b) = (whole.feed(t).unwrap(), cut.feed(t).unwrap());
            let (vs_whole, vs_oracle) = (
                max_rel_err(&b, &a),
                max_rel_err(&b, &want[i * 17..(i + 1) * 17]),
            );
            assert!(
                vs_whole < FORWARD && vs_oracle < FORWARD,
                "biases {biases} pos {i}: {vs_whole} {vs_oracle}"
            );
        }
    }
}

#[test]
fn sharded_prefill_with_many_rows_equals_the_whole_model() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 5, true, false);
    let whole = GpuLlama::new(&g, &w, None, None).unwrap();
    let cut = GpuLlama::with_binding_limit(&g, &w, None, None, LIMIT).unwrap();
    let e = max_rel_err(
        &cut.forward_all(&IDS).unwrap(),
        &whole.forward_all(&IDS).unwrap(),
    );
    assert!(e < FORWARD, "{e}");
    // scoring goes through the same path
    let (a, b) = (whole.score(&IDS).unwrap(), cut.score(&IDS).unwrap());
    assert_eq!(
        a.iter().map(|x| x.1).collect::<Vec<_>>(),
        b.iter().map(|x| x.1).collect::<Vec<_>>()
    );
}

#[test]
fn sharded_quantised_models_equal_their_whole_quantised_selves() {
    let Some(g) = open_for_test() else { return };
    for kind in [QKind::F16, QKind::Q8_0, QKind::Q4_0] {
        let w = Llama::random(cfg(), 3, false, false);
        let whole = &mut GpuLlama::new(&g, &w, Some(kind), None).unwrap();
        let cut = &mut GpuLlama::with_binding_limit(&g, &w, Some(kind), None, LIMIT).unwrap();
        for &t in &IDS {
            let e = max_rel_err(&cut.feed(t).unwrap(), &whole.feed(t).unwrap());
            assert!(e < FORWARD, "{kind:?}: {e}");
        }
    }
}

#[test]
fn the_plan_lists_the_blocks_and_keeps_the_total() {
    let c = cfg();
    let whole = plan(&c, None);
    let cut = plan_limited(&c, None, LIMIT);
    assert_eq!(
        whole.total(),
        cut.total(),
        "splitting must not change the bytes"
    );
    assert!(
        cut.rows().iter().any(|r| r.0 == "tok_emb.0") && cut.rows().iter().any(|r| r.0 == "head.1")
    );
    assert!(
        cut.rows().iter().all(|r| r.1 <= LIMIT),
        "{:?}",
        cut.rows().iter().find(|r| r.1 > LIMIT)
    );
    cut.check(LIMIT, None).unwrap();
    // no limit means no change
    assert_eq!(plan_limited(&c, None, u64::MAX).rows(), whole.rows());
}
