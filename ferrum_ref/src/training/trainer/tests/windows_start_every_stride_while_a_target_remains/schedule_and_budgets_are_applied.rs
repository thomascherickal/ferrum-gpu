use super::*;

#[test]
fn schedule_and_budgets_are_applied() {
    let toks = corpus(200);
    let mut m = Mock::default();
    let c = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 10,
        lr: 1.0,
        cosine: true,
        warmup: 2,
        ..TrainConfig::default()
    };
    let r = train(&mut m, &toks, &c, None, None, &mut NoHooks).unwrap();
    assert_eq!(r.steps_run, 10);
    let lrs: Vec<f32> = m.calls.iter().map(|c| c.3).collect();
    assert!(
        (lrs[0] - 0.5).abs() < 1e-6 && (lrs[1] - 1.0).abs() < 1e-6,
        "linear warmup: {lrs:?}"
    );
    assert!(
        lrs.windows(2).skip(1).all(|w| w[1] <= w[0] + 1e-6),
        "then cosine decay: {lrs:?}"
    );
    assert_eq!(*lrs.last().unwrap(), 0.0, "ends at zero");
    // Without cosine the rate is fixed; weight decay is passed through.
    let mut m2 = Mock::default();
    let c2 = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 3,
        lr: 0.25,
        ..TrainConfig::default()
    };
    train(&mut m2, &toks, &c2, None, None, &mut NoHooks).unwrap();
    assert!(m2.calls.iter().all(|c| c.3 == 0.25));
}

#[test]
fn epochs_run_to_completion_across_epoch_boundaries() {
    let toks = corpus(100); // 12 windows at context 8
    let mut m = Mock::default();
    let c = TrainConfig {
        context: 8,
        batch: 5,
        epochs: 2,
        ..TrainConfig::default()
    };
    let r = train(&mut m, &toks, &c, None, None, &mut NoHooks).unwrap();
    assert_eq!(r.steps_run, 6, "3 steps per epoch x 2");
    assert_eq!(r.final_step, 6);
    assert!(!r.stopped);
    let ep0: Vec<usize> = m.calls[..3]
        .iter()
        .flat_map(|c| c.0.chunks(8).map(|w| w[0] * 1000 + w[1]))
        .collect();
    let ep1: Vec<usize> = m.calls[3..]
        .iter()
        .flat_map(|c| c.0.chunks(8).map(|w| w[0] * 1000 + w[1]))
        .collect();
    assert_eq!(ep0.len(), 12);
    assert_eq!(ep1.len(), 12);
}
