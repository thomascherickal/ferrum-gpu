use super::*;

#[test]
pub(super) fn token_embedding_gradient_sums_repeated_tokens() {
    let m = Llama::random(tiny(RopeType::Norm), 26, false, false);
    let ids = vec![2usize, 2, 2, 5];
    let (_, g) = loss_and_grads(&m, &ids, &[1, 3, 4, 0], 1, 1).unwrap();
    let d = m.cfg.dim;
    assert!(g.tok[2 * d..3 * d].iter().any(|&v| v != 0.0));
    assert!(
        g.tok[..d].iter().all(|&v| v == 0.0),
        "an unused token has no gradient"
    );
}

#[test]
pub(super) fn batching_equals_two_separate_sequences_and_threads_do_not_matter() {
    let m = Llama::random(tiny(RopeType::Neox), 27, true, false);
    let (ids, targets) = data(&m.cfg, 2, 3);
    let (l, g) = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    let (l0, g0) = loss_and_grads(&m, &ids[..3], &targets[..3], 1, 1).unwrap();
    let (l1, g1) = loss_and_grads(&m, &ids[3..], &targets[3..], 1, 1).unwrap();
    assert!((l - (l0 + l1) / 2.0).abs() < 1e-5);
    for (pi, p) in g.params().iter().enumerate() {
        for (j, &v) in p.iter().enumerate() {
            let want = (g0.params()[pi][j] + g1.params()[pi][j]) / 2.0;
            assert!((v - want).abs() < 1e-5, "param {pi}[{j}]");
        }
    }
    let (l4, g4) = loss_and_grads(&m, &ids, &targets, 2, 4).unwrap();
    assert!((l - l4).abs() < 1e-6);
    assert!(g
        .params()
        .iter()
        .zip(g4.params())
        .all(|(a, b)| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)));
}

#[test]
pub(super) fn loss_matches_the_forward_oracle_and_bad_input_is_refused() {
    let m = Llama::random(tiny(RopeType::Norm), 28, false, false);
    let (ids, targets) = data(&m.cfg, 1, 4);
    let (loss, _) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
    let want = nn::cross_entropy(&m.forward_logits(&ids).unwrap(), &targets, 7).0;
    assert!((loss - want).abs() < 1e-5, "{loss} vs {want}");
    let e = |r: Result<(f32, Llama), String>| r.expect_err("must refuse");
    assert!(e(loss_and_grads(&m, &ids, &targets, 3, 1)).contains("split"));
    assert!(e(loss_and_grads(&m, &[0; 6], &[0; 6], 1, 1)).contains("context"));
    assert!(e(loss_and_grads(&m, &ids, &targets[..3], 1, 1)).contains("targets"));
    assert!(e(loss_and_grads(&m, &[7, 0, 0, 0], &targets, 1, 1)).contains("out of range"));
    let tied = Llama::random(tiny(RopeType::Norm), 29, false, true);
    assert!(e(loss_and_grads(&tied, &ids, &targets, 1, 1)).contains("tied"));
}

#[test]
pub(super) fn param_accessors_agree_with_each_other_and_with_decay_rules() {
    let m = Llama::random(tiny(RopeType::Neox), 30, true, false);
    let n = m.params().len();
    assert_eq!(n, 1 + 2 * BLOCK_PARAMS + 2);
    assert_eq!(
        m.params().iter().map(|p| p.len()).sum::<usize>(),
        m.num_params()
    );
    let z = m.zeros_like();
    assert!(z.params().iter().all(|p| p.iter().all(|&v| v == 0.0)));
    assert_eq!(
        z.params().iter().map(|p| p.len()).collect::<Vec<_>>(),
        m.params().iter().map(|p| p.len()).collect::<Vec<_>>()
    );
    let decayed: Vec<bool> = (0..n).map(|i| m.decays(i)).collect();
    assert!(decayed[0] && decayed[n - 1], "embedding and head decay");
    assert!(!decayed[n - 2], "output norm does not");
    // per block: norms and biases do not decay, matrices do
    for blk in 0..2 {
        let o = 1 + blk * BLOCK_PARAMS;
        let want = [
            false, true, false, true, false, true, false, true, false, true, true, true,
        ];
        assert_eq!(&decayed[o..o + BLOCK_PARAMS], &want);
    }
    // every decayed parameter is a matrix (more than a vector's worth of elements per row)
    assert_eq!(m.params()[1 + 1].len(), 8 * 8); // wq of block 0
}
