use super::*;
use crate::gpt2::Config;
use crate::grad::loss_and_grads;

fn cfg() -> Config {
    Config {
        vocab: 9,
        ctx: 6,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 12,
    }
}

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

#[test]
fn schedule_warms_up_peaks_and_decays_to_zero() {
    let s = Schedule {
        base_lr: 1.0,
        warmup: 10,
        total: 110,
    };
    assert!((s.lr_at(1) - 0.1).abs() < 1e-6);
    assert!((s.lr_at(10) - 1.0).abs() < 1e-6);
    assert!(
        (s.lr_at(60) - 0.5).abs() < 1e-5,
        "halfway through the cosine"
    );
    assert_eq!(s.lr_at(110), 0.0);
    assert_eq!(s.lr_at(5000), 0.0);
    assert_eq!(s.lr_at(0), s.lr_at(1), "step 0 is treated as step 1");
    let mut prev = f32::MAX;
    for step in 10..=110 {
        let lr = s.lr_at(step);
        assert!(lr <= prev + 1e-7, "monotone decay at {step}");
        prev = lr;
    }
    assert_eq!(
        Schedule {
            base_lr: 2.0,
            warmup: 0,
            total: 10
        }
        .lr_at(1),
        2.0 * 0.5 * (1.0 + (0.1f32 * std::f32::consts::PI).cos())
    );
}

#[test]
fn training_a_tiny_model_on_one_batch_drives_the_loss_down() {
    let c = cfg();
    let mut w = Gpt2::init(c, 6);
    let ids: Vec<usize> = (0..12).map(|i| (i * 2 + 1) % 9).collect();
    let targets: Vec<usize> = ids.iter().map(|&i| (i + 3) % 9).collect();
    let mut st = AdamState::new(&w);
    let first = loss_and_grads(&w, &ids, &targets, 2, 1).unwrap().0;
    for _ in 0..60 {
        let (_, g) = loss_and_grads(&w, &ids, &targets, 2, 1).unwrap();
        adam_step(&mut w, &g, &mut st, &Adam::new(0.01), 1.0);
    }
    let last = loss_and_grads(&w, &ids, &targets, 2, 1).unwrap().0;
    assert!(first > 2.0, "starts near ln(9) = 2.2, got {first}");
    assert!(last < first * 0.3, "memorises one batch: {first} -> {last}");
}

#[test]
fn bias_corrections_approach_one() {
    let a = Adam::new(0.1);
    let (b1, b2) = a.bias_corrections(1);
    assert!((b1 - 0.1).abs() < 1e-6 && (b2 - 0.001).abs() < 1e-6);
    let (b1, b2) = a.bias_corrections(100_000);
    assert!((b1 - 1.0).abs() < 1e-6 && (b2 - 1.0).abs() < 1e-3);
    assert!(a.bias_corrections(u64::MAX).0.is_finite());
}
