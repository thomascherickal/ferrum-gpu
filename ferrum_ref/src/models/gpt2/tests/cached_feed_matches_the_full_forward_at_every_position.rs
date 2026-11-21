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
