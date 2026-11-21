use super::*;

#[test]
fn token_embedding_gradient_sums_repeated_tokens() {
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
fn batching_equals_two_separate_sequences_and_threads_do_not_matter() {
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
fn loss_matches_the_forward_oracle_and_bad_input_is_refused() {
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
fn param_accessors_agree_with_each_other_and_with_decay_rules() {
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

#[test]
fn adam_step_matches_the_hand_computed_first_update_and_clipping_caps_the_norm() {
    let mut m = Llama::random(tiny(RopeType::Norm), 31, false, false);
    let mut g = m.zeros_like();
    g.out_norm.fill(2.0);
    g.head[0] = -3.0;
    let before = m.clone();
    let mut st = Moments::new(&m);
    let adam = Adam {
        weight_decay: 0.1,
        ..Adam::new(0.01)
    };
    adam_step(&mut m, &g, &mut st, &adam, 1.0);
    // step 1: m_hat = g, v_hat = g^2, so the update is lr * g / (|g| + eps) = lr * sign(g)
    assert!(
        (m.out_norm[0] - (before.out_norm[0] - 0.01)).abs() < 1e-6,
        "no decay on the norm gain"
    );
    let want = before.head[0] + 0.01 - 0.01 * 0.1 * before.head[0];
    assert!(
        (m.head[0] - want).abs() < 1e-6,
        "decayed head: {} vs {want}",
        m.head[0]
    );
    // zero gradient: only the decoupled decay moves a decayed tensor, by lr * wd * p
    for (a, b) in m.tok.iter().zip(&before.tok) {
        assert!((a - (b - 0.01 * 0.1 * b)).abs() < 1e-7);
    }
    assert_eq!(
        m.blocks[0].attn_norm, before.blocks[0].attn_norm,
        "norm gains never decay"
    );
    assert_eq!(st.t, 1);
    assert!((global_norm(&g) - (8.0f32 * 4.0 + 9.0).sqrt()).abs() < 1e-5);
}

#[test]
fn training_a_fixed_batch_drives_the_loss_down() {
    let m = Llama::random(tiny(RopeType::Norm), 32, false, false);
    let (ids, targets) = data(&m.cfg, 2, 4);
    let mut tr = LlamaTrainer::new(m, 1);
    let sched = Schedule {
        base_lr: 0.05,
        warmup: 2,
        total: 40,
    };
    let mut losses = Vec::new();
    for step in 1..=40u64 {
        let cfg = StepConfig {
            adam: Adam::new(sched.lr_at(step)),
            clip: 1.0,
        };
        losses.push(tr.step(&ids, &targets, 2, &cfg).unwrap().loss);
    }
    assert_eq!(tr.steps(), 40);
    assert!(
        losses[39] < 0.3 * losses[0],
        "loss {} -> {}",
        losses[0],
        losses[39]
    );
    assert!(losses.iter().all(|l| l.is_finite()));
}
