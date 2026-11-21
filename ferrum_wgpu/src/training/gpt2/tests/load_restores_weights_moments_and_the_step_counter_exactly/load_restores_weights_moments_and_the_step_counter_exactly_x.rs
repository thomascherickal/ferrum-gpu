use super::*;

#[test]
fn load_restores_weights_moments_and_the_step_counter_exactly() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let (ids, targets) = batch_data(&c, 2, 4, 1);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    // Train 3 steps straight through.
    let mut a = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
    for _ in 0..3 {
        a.step(&ids, &targets, 2, &sc).unwrap();
    }
    // Train 2, checkpoint to the host, resume in a fresh trainer for the third.
    let mut b = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
    for _ in 0..2 {
        b.step(&ids, &targets, 2, &sc).unwrap();
    }
    let (w, (m, v), t) = (
        b.weights_host().unwrap(),
        b.moments_host().unwrap(),
        b.steps(),
    );
    let mut resumed = GpuTrainer::new(&g, &Gpt2::init(c, 99), None).unwrap();
    resumed.load(&w, Some((&m, &v)), t).unwrap();
    resumed.step(&ids, &targets, 2, &sc).unwrap();
    assert_eq!(resumed.steps(), 3);
    assert_eq!(
        resumed.weights_host().unwrap(),
        a.weights_host().unwrap(),
        "resume = uninterrupted"
    );
    assert_eq!(resumed.moments_host().unwrap(), a.moments_host().unwrap());
    // A different shape is refused.
    let other = Gpt2::init(Config { dim: 12, ..c }, 1);
    assert!(resumed
        .load(&other, None, 0)
        .unwrap_err()
        .to_string()
        .contains("shape"));
}
