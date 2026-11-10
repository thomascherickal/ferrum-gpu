use super::*;

#[test]
fn cached_feed_matches_the_full_forward_at_every_position() {
    let w = Gpt2::random(tiny(), 9);
    let ids = [4usize, 0, 10, 3, 3, 7, 1, 2]; // fills the context
    let want = w.forward_logits(&ids).unwrap();
    let mut cache = Cache::new(w.cfg.layers);
    assert!(cache.is_empty());
    for (i, &t) in ids.iter().enumerate() {
        let got = w.feed(&mut cache, t).unwrap();
        let row = &want[i * 11..(i + 1) * 11];
        assert!(
            got.iter()
                .zip(row)
                .all(|(a, b)| (a - b).abs() < 1e-5 * b.abs().max(1.0)),
            "position {i}"
        );
    }
    assert_eq!(cache.len(8), 8);
    assert!(w.feed(&mut cache, 1).unwrap_err().contains("full"));
    cache.clear();
    assert!(cache.is_empty());
    assert!(w.feed(&mut cache, 11).unwrap_err().contains("out of range"));
}

pub(crate) fn tiny() -> Config {
    Config {
        vocab: 11,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 2,
        hidden: 12,
    }
}

#[test]
fn params_are_in_canonical_order_and_counted() {
    let w = Gpt2::random(tiny(), 1);
    let p = w.params();
    assert_eq!(p.len(), 2 + 2 * 16 + 3);
    assert!(std::ptr::eq(p[0], &w.tok) && std::ptr::eq(p[1], &w.pos));
    assert!(std::ptr::eq(p[2], &w.blocks[0].ln1_g));
    assert!(std::ptr::eq(p[2 + 16 + 8], &w.blocks[1].wo));
    assert!(std::ptr::eq(p[p.len() - 1], &w.head));
    assert_eq!(w.num_params(), p.iter().map(|x| x.len()).sum::<usize>());
    let mut z = w.zeros_like();
    assert!(z.params().iter().all(|p| p.iter().all(|&v| v == 0.0)));
    assert_eq!(z.cfg, w.cfg);
    z.params_mut()[0][0] = 5.0;
    assert_eq!(z.tok[0], 5.0);
}

#[test]
fn param_lens_agree_with_a_real_model_and_from_params_round_trips() {
    let w = Gpt2::random(tiny(), 5);
    let lens = param_lens(&tiny()).unwrap();
    assert_eq!(lens, w.params().iter().map(|p| p.len()).collect::<Vec<_>>());
    let back = Gpt2::from_params(tiny(), w.params().into_iter().cloned().collect()).unwrap();
    assert_eq!(back, w);
    // Wrong count, wrong length, bad config.
    let mut short: Vec<Vec<f32>> = w.params().into_iter().cloned().collect();
    short.pop();
    assert!(Gpt2::from_params(tiny(), short).is_err());
    let mut bent: Vec<Vec<f32>> = w.params().into_iter().cloned().collect();
    bent[3].push(0.0);
    assert!(Gpt2::from_params(tiny(), bent)
        .unwrap_err()
        .contains("do not match"));
    assert!(Gpt2::from_params(Config { heads: 3, ..tiny() }, vec![]).is_err());
}

#[test]
fn param_lens_refuses_overflowing_and_invalid_configs() {
    assert!(param_lens(&Config {
        vocab: usize::MAX,
        ..tiny()
    })
    .is_none());
    assert!(param_lens(&Config {
        dim: usize::MAX / 2 + 2,
        heads: 1,
        ..tiny()
    })
    .is_none());
    assert!(param_lens(&Config {
        layers: 0,
        ..tiny()
    })
    .is_none());
}

#[test]
fn decay_flags_match_the_matrix_rule() {
    let w = Gpt2::random(tiny(), 1);
    let flags: Vec<bool> = (0..w.params().len()).map(|i| w.decays(i)).collect();
    assert!(flags[0] && flags[1], "embeddings decay");
    assert!(!flags[2] && !flags[3], "ln1 gain/shift do not");
    assert!(flags[4] && !flags[5], "wq decays, bq does not");
    assert!(flags[2 + 12] && !flags[2 + 13], "w1 decays, b1 does not");
    let n = flags.len();
    assert!(!flags[n - 3] && !flags[n - 2], "final LayerNorm does not");
    assert!(flags[n - 1], "head decays");
    // Cross-check against the shapes: decayed tensors are exactly the ones longer than dim.
    for (i, p) in w.params().iter().enumerate() {
        assert_eq!(
            flags[i],
            p.len() > w.cfg.hidden.max(w.cfg.dim),
            "param {i} len {}",
            p.len()
        );
    }
}

#[test]
fn init_follows_the_gpt_recipe() {
    let w = Gpt2::init(tiny(), 3);
    assert!(w.blocks[0].ln1_g.iter().all(|&g| g == 1.0));
    assert!(w.lnf_g.iter().all(|&g| g == 1.0));
    assert!(w.blocks[0].bq.iter().chain(&w.lnf_b).all(|&b| b == 0.0));
    let std = (w.tok.iter().map(|v| v * v).sum::<f32>() / w.tok.len() as f32).sqrt();
    assert!((std - 0.02).abs() < 0.005, "std {std}");
    assert_eq!(w, Gpt2::init(tiny(), 3));
}

#[test]
fn config_validation() {
    assert!(tiny().validate().is_ok());
    assert!(Config { heads: 3, ..tiny() }
        .validate()
        .unwrap_err()
        .contains("divisible"));
    assert!(Config {
        layers: 0,
        ..tiny()
    }
    .validate()
    .unwrap_err()
    .contains("positive"));
    assert_eq!(tiny().head_dim(), 4);
}

#[test]
fn random_weights_have_the_right_shapes_and_are_deterministic() {
    let w = Gpt2::random(tiny(), 1);
    assert_eq!(w.tok.len(), 11 * 8);
    assert_eq!(w.pos.len(), 8 * 8);
    assert_eq!(w.head.len(), 8 * 11);
    assert_eq!(w.blocks.len(), 2);
    assert_eq!(w.blocks[0].w1.len(), 8 * 12);
    assert_eq!(w, Gpt2::random(tiny(), 1));
    assert_ne!(w, Gpt2::random(tiny(), 2));
}

#[test]
fn forward_is_causal_prefix_stable() {
    // Logits for the first rows must not depend on later tokens.
    let w = Gpt2::random(tiny(), 3);
    let full = w.forward_logits(&[1, 5, 2, 9, 4]).unwrap();
    let prefix = w.forward_logits(&[1, 5, 2]).unwrap();
    assert_eq!(full.len(), 5 * 11);
    assert_eq!(&full[..3 * 11], &prefix[..]);
    assert!(full.iter().all(|v| v.is_finite()));
}

#[test]
fn forward_depends_on_the_tokens() {
    let w = Gpt2::random(tiny(), 3);
    assert_ne!(
        w.forward_logits(&[1, 2]).unwrap(),
        w.forward_logits(&[1, 3]).unwrap()
    );
}

#[test]
fn forward_rejects_bad_input() {
    let w = Gpt2::random(tiny(), 3);
    assert!(w
        .forward_logits(&[])
        .unwrap_err()
        .contains("sequence length"));
    assert!(w
        .forward_logits(&[0; 9])
        .unwrap_err()
        .contains("sequence length"));
    assert!(w
        .forward_logits(&[11])
        .unwrap_err()
        .contains("out of range"));
}
