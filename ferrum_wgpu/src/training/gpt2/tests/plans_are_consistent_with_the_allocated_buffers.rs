use super::*;

#[test]
fn plans_are_consistent_with_the_allocated_buffers() {
    let c = cfg();
    let sp = static_plan(&c);
    assert_eq!(sp.rows().len(), 4 * (2 + 2 * 16 + 3));
    let w = Gpt2::random(c, 1);
    assert_eq!(sp.total(), 4 * 4 * w.num_params() as u64);
    let ap = activation_plan(&c, 2, 5);
    assert!(ap.rows().iter().any(|(n, _)| n == "blk1.probs"));
    assert!(activation_plan(&c, 4, 5).total() > ap.total());
}

#[test]
fn load_refuses_moments_of_another_shape() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 31);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let other = Gpt2::init(Config { dim: 12, ..c }, 31);
    let e = tr
        .load(&w, Some((&other.zeros_like(), &w.zeros_like())), 0)
        .unwrap_err();
    assert!(e.to_string().contains("moment shapes differ"), "{e}");
    let e = tr
        .load(&w, Some((&w.zeros_like(), &other.zeros_like())), 0)
        .unwrap_err();
    assert!(e.to_string().contains("moment shapes differ"), "{e}");
    assert_eq!(tr.steps(), 0, "a refused load changes nothing");
}

#[test]
fn a_gpu_fault_at_any_command_fails_the_step_cleanly_and_nothing_is_applied() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 41);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let (ids, targets) = batch_data(&c, 2, 4, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    };
    let mut attempts = 0;
    let commands = g.sweep_faults(|| {
        attempts += 1;
        let before = tr.steps();
        let r = tr.step(&ids, &targets, 2, &sc).map(|_| ());
        if r.is_err() {
            assert_eq!(tr.steps(), before, "a failed step does not count");
        }
        r
    });
    // The failed attempts discarded their batches, so only the final success touched the weights.
    assert!(
        commands > 100,
        "a training step issues many commands, got {commands}"
    );
    assert_eq!(attempts, commands + 1);
    assert_eq!(tr.steps(), 1);
    let fresh = GpuTrainer::new(&g, &w, None).unwrap();
    assert_eq!(
        fresh.weights_host().unwrap(),
        w,
        "sanity: the baseline weights are what we started from"
    );
    assert_ne!(
        tr.weights_host().unwrap(),
        w,
        "the one successful step updated them"
    );
}

#[test]
fn loss_and_grads_reports_a_fault_too() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 42), None).unwrap();
    let (ids, targets) = batch_data(&c, 1, 3, 0);
    let commands = g.sweep_faults(|| tr.loss_and_grads(&ids, &targets, 1).map(|_| ()));
    assert!(commands > 50, "{commands}");
}
