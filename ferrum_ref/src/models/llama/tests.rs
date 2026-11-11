use super::params::transpose;
use super::*;

pub fn cfg() -> Config {
    Config {
        vocab: 13,
        ctx: 12,
        dim: 8,
        heads: 4,
        kv_heads: 2,
        head_dim: 4,
        layers: 2,
        hidden: 16,
        rope_dim: 4,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn config_validation_names_each_problem() {
    assert!(cfg().validate().is_ok());
    let bad = |f: &dyn Fn(&mut Config)| {
        let mut c = cfg();
        f(&mut c);
        c.validate().unwrap_err()
    };
    assert!(bad(&|c| c.vocab = 0).contains("positive"));
    assert!(bad(&|c| c.kv_heads = 3).contains("multiple"));
    assert!(bad(&|c| c.rope_dim = 5).contains("rope_dim"));
    assert!(bad(&|c| c.rope_dim = 6).contains("rope_dim"));
    assert!(bad(&|c| c.eps = 0.0).contains("eps"));
    assert!(bad(&|c| c.rope_base = f32::NAN).contains("eps"));
    assert_eq!((cfg().q_dim(), cfg().kv_dim()), (16, 8));
}

#[test]
fn head_dim_may_differ_from_dim_over_heads() {
    let mut c = cfg();
    c.head_dim = 6;
    c.rope_dim = 6;
    let m = Llama::random(c, 1, false, false);
    assert_eq!(m.blocks[0].wq.len(), 8 * 24);
    assert_eq!(m.blocks[0].wo.len(), 24 * 8);
    let logits = m.forward_logits(&[1, 2, 3]).unwrap();
    assert_eq!(logits.len(), 3 * 13);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn param_lens_matches_the_random_model_and_refuses_hostile_configs() {
    let m = Llama::random(cfg(), 1, false, false);
    let lens = param_lens(&m.cfg).unwrap();
    assert_eq!(lens.iter().sum::<usize>(), m.num_params());
    let mut c = cfg();
    c.kv_heads = 3;
    assert!(param_lens(&c).is_none());
    let mut c = cfg();
    c.vocab = usize::MAX;
    c.dim = usize::MAX;
    assert!(param_lens(&c).is_none(), "overflow must not allocate");
}

#[test]
fn tied_head_is_the_transposed_embedding_and_is_not_counted_twice() {
    let t = Llama::random(cfg(), 2, false, true);
    assert_eq!(t.head, transpose(&t.tok, 13, 8));
    let u = Llama::random(cfg(), 2, false, false);
    assert_eq!(t.num_params() + 13 * 8, u.num_params());
}

#[test]
fn cached_decode_matches_the_full_forward_for_every_variant() {
    for (ty, biases, kv) in [
        (RopeType::Norm, false, 2),
        (RopeType::Neox, true, 2),
        (RopeType::Norm, false, 4),
        (RopeType::Neox, true, 1),
    ] {
        let mut c = cfg();
        c.rope_type = ty;
        c.kv_heads = kv;
        let m = Llama::random(c, 3, biases, false);
        let ids = [3, 1, 4, 1, 5, 9, 2, 6];
        let full = m.forward_logits(&ids).unwrap();
        let mut cache = Cache::new(c.layers);
        assert!(cache.is_empty());
        for (i, &id) in ids.iter().enumerate() {
            let step = m.feed(&mut cache, id).unwrap();
            let want = &full[i * 13..(i + 1) * 13];
            assert!(
                max_diff(&step, want) < 1e-4,
                "{ty:?} bias={biases} kv={kv} pos {i}"
            );
        }
        assert_eq!(cache.len(c.kv_dim()), ids.len());
    }
}

#[test]
fn a_prefix_gives_the_same_logits_as_the_prefix_of_a_longer_run() {
    let m = Llama::random(cfg(), 4, false, false);
    let long = m.forward_logits(&[1, 2, 3, 4, 5]).unwrap();
    let short = m.forward_logits(&[1, 2, 3]).unwrap();
    assert!(max_diff(&long[..3 * 13], &short) < 1e-5, "causality");
}

#[test]
fn rope_scaling_changes_the_output_and_unit_scaling_does_not() {
    let mut m = Llama::random(cfg(), 5, false, false);
    let ids = [1, 2, 3, 4];
    let base = m.forward_logits(&ids).unwrap();
    m.rope = RopeScaling {
        freq_factors: vec![1.0, 1.0],
        linear_factor: 1.0,
    };
    assert_eq!(m.forward_logits(&ids).unwrap(), base);
    m.rope.linear_factor = 4.0;
    assert!(max_diff(&m.forward_logits(&ids).unwrap(), &base) > 1e-4);
}

#[test]
fn bad_input_is_refused_with_the_reason() {
    let m = Llama::random(cfg(), 6, false, false);
    assert!(m
        .forward_logits(&[])
        .unwrap_err()
        .contains("sequence length"));
    assert!(m
        .forward_logits(&[0; 13])
        .unwrap_err()
        .contains("sequence length"));
    assert!(m
        .forward_logits(&[13])
        .unwrap_err()
        .contains("out of range"));
    let mut cache = Cache::new(2);
    for _ in 0..12 {
        m.feed(&mut cache, 1).unwrap();
    }
    assert!(m.feed(&mut cache, 1).unwrap_err().contains("full"));
    assert!(m.feed(&mut Cache::new(2), 99).is_err());
}

#[test]
fn biases_are_applied_to_q_k_and_v() {
    let a = Llama::random(cfg(), 7, true, false);
    let mut b = a.clone();
    for blk in &mut b.blocks {
        blk.bv.iter_mut().for_each(|v| *v += 1.0);
    }
    assert!(
        max_diff(
            &a.forward_logits(&[1, 2]).unwrap(),
            &b.forward_logits(&[1, 2]).unwrap()
        ) > 1e-3
    );
}
