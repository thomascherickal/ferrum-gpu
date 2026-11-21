use super::*;

#[test]
pub(super) fn adam_step_matches_the_hand_computed_first_update_and_clipping_caps_the_norm() {
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
    let head = m.head[0];
    assert!((head - want).abs() < 1e-6, "decayed head: {head} vs {want}");
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
pub(super) fn training_a_fixed_batch_drives_the_loss_down() {
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
    let (first, last) = (losses[0], losses[39]);
    assert!(last < 0.3 * first, "loss {first} -> {last}");
    assert!(losses.iter().all(|l| l.is_finite()));
}
