//! G2: a model whose tensors are larger than the (lowered) binding limit is stored as blocks and
//! must give the same answers as the same weights stored whole.

use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, FORWARD};
use ferrum_ref::llama::RopeType;

fn cfg() -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        layers: 2,
        hidden: 64,
        rope_dim: 8,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

/// Larger than the biggest buffer that cannot be split (the KV cache is 640 bytes), smaller than
/// every weight: the embedding (2176 bytes) and each matrix (4096 to 8192) are cut into blocks.
const LIMIT: u64 = 1000;
const IDS: [usize; 10] = [3, 16, 0, 9, 9, 4, 11, 2, 7, 5];

fn blocks_of(m: &GpuLlama) -> (usize, usize, usize) {
    let head = match &m.head {
        Lin::Sharded { parts, .. } => parts.len(),
        _ => 1,
    };
    let gate = match &m.blocks[0].gate {
        Lin::Sharded { parts, .. } => parts.len(),
        _ => 1,
    };
    (m.tok.parts(), head, gate)
}

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
