use super::*;

#[test]
fn first_step_moves_each_parameter_by_about_lr() {
    // With bias-corrected moments the first Adam step is lr * sign(g) (up to eps).
    let mut w = Gpt2::random(cfg(), 1);
    let before = w.clone();
    let mut g = w.zeros_like();
    g.params_mut().into_iter().for_each(|p| {
        p.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = if i % 2 == 0 { 0.5 } else { -0.25 })
    });
    let mut st = AdamState::new(&w);
    adam_step(&mut w, &g, &mut st, &Adam::new(0.01), 1.0);
    assert_eq!(st.t, 1);
    assert!((w.tok[0] - (before.tok[0] - 0.01)).abs() < 1e-6);
    assert!((w.tok[1] - (before.tok[1] + 0.01)).abs() < 1e-6);
}

#[test]
fn weight_decay_skips_biases_and_layernorm_but_not_matrices() {
    let mut w = Gpt2::random(cfg(), 2);
    let before = w.clone();
    let g = w.zeros_like(); // zero gradient: only decay can move anything
    let mut st = AdamState::new(&w);
    let adam = Adam {
        weight_decay: 0.1,
        ..Adam::new(0.5)
    };
    adam_step(&mut w, &g, &mut st, &adam, 1.0);
    assert_eq!(w.blocks[0].bq, before.blocks[0].bq, "bias untouched");
    assert_eq!(
        w.blocks[0].ln1_g, before.blocks[0].ln1_g,
        "LayerNorm untouched"
    );
    assert_eq!(w.lnf_b, before.lnf_b);
    let expect = before.blocks[0].wq[3] - 0.5 * 0.1 * before.blocks[0].wq[3];
    assert!(
        (w.blocks[0].wq[3] - expect).abs() < 1e-6,
        "matrix decays by lr*wd*w"
    );
    assert!(w.head[0] != before.head[0] && w.tok[0] != before.tok[0]);
}

#[test]
fn clipping_caps_the_global_norm_and_reports_the_original() {
    let w = Gpt2::random(cfg(), 3);
    let mut g = w.clone();
    let n0 = global_norm(&g);
    assert!(n0 > 1.0);
    let reported = clip_grad_norm(&mut g, 1.0);
    assert_eq!(reported, n0);
    assert!((global_norm(&g) - 1.0).abs() < 1e-4);
    let mut small = w.zeros_like();
    small.tok[0] = 0.1;
    clip_grad_norm(&mut small, 1.0);
    assert_eq!(small.tok[0], 0.1, "a norm within the cap is untouched");
    assert_eq!(clip_scale(f32::NAN, 1.0), 1.0);
    assert_eq!(clip_scale(f32::INFINITY, 1.0), 1.0);
    assert_eq!(clip_scale(5.0, 0.0), 1.0);
    assert_eq!(clip_scale(4.0, 2.0), 0.5);
}

#[test]
fn grad_scale_is_the_same_as_clipping_first() {
    let w0 = Gpt2::random(cfg(), 4);
    let mut g = Gpt2::random(cfg(), 5);
    let mut a = w0.clone();
    let mut b = w0.clone();
    let mut ga = g.clone();
    let scale = clip_scale(global_norm(&g), 0.5);
    let adam = Adam {
        weight_decay: 0.01,
        ..Adam::new(0.01)
    };
    adam_step(&mut a, &g, &mut AdamState::new(&w0), &adam, scale);
    clip_grad_norm(&mut ga, 0.5);
    adam_step(&mut b, &ga, &mut AdamState::new(&w0), &adam, 1.0);
    assert_eq!(a, b);
    g.tok[0] = 0.0;
}
