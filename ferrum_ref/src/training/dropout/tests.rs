use super::*;

#[test]
fn a_rate_of_zero_or_less_is_no_dropout() {
    assert_eq!(Dropout::new(0.0, 1), None);
    assert_eq!(Dropout::new(-0.5, 1), None);
    assert!(Dropout::new(0.1, 1).is_some());
}

#[test]
fn the_rate_is_clamped_to_the_maximum() {
    assert_eq!(Dropout::new(0.99, 1).unwrap().rate, MAX_RATE);
    assert_eq!(Dropout::new(0.3, 1).unwrap().rate, 0.3);
}

#[test]
fn survivors_are_scaled_so_the_expectation_is_unchanged() {
    let d = Dropout::new(0.25, 7).unwrap();
    let m = d.mask(0, 40_000);
    let scale = 1.0 / 0.75;
    assert!(m.iter().all(|&x| x == 0.0 || x == scale));
    let dropped = m.iter().filter(|&&x| x == 0.0).count() as f32 / m.len() as f32;
    assert!((dropped - 0.25).abs() < 0.01, "dropped fraction {dropped}");
    let mean = m.iter().sum::<f32>() / m.len() as f32;
    assert!((mean - 1.0).abs() < 0.02, "mean multiplier {mean}");
}

#[test]
fn a_mask_is_a_pure_function_of_seed_block_and_index() {
    let d = Dropout::new(0.3, 42).unwrap();
    assert_eq!(d.mask(2, 500), d.mask(2, 500));
    // a longer mask starts with the shorter one: index i does not depend on n
    assert_eq!(d.mask(2, 800)[..500], d.mask(2, 500)[..]);
}

#[test]
fn seed_and_block_each_change_the_pattern() {
    let d = Dropout::new(0.3, 42).unwrap();
    let other_seed = Dropout::new(0.3, 43).unwrap();
    assert_ne!(d.mask(0, 500), other_seed.mask(0, 500));
    assert_ne!(d.mask(0, 500), d.mask(1, 500));
}

#[test]
fn step_seeds_differ_per_step_and_per_run() {
    let seeds: Vec<u64> = (0..100).map(|s| step_seed(5, s)).collect();
    let mut sorted = seeds.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 100, "no repeats across 100 steps");
    assert_ne!(step_seed(5, 3), step_seed(6, 3));
    assert_eq!(step_seed(5, 3), step_seed(5, 3));
}

#[test]
fn uniform_values_stay_in_the_unit_interval() {
    for i in 0..10_000 {
        let u = uniform(9, 1, 3, i);
        assert!((0.0..1.0).contains(&u), "{u}");
    }
}
