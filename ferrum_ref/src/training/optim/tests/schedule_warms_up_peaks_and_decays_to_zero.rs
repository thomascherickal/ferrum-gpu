use super::*;

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
