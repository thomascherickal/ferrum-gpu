use super::*;

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
