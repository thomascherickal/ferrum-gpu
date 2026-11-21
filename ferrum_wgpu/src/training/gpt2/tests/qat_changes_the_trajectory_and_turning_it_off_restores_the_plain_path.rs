use super::*;

#[test]
fn qat_changes_the_trajectory_and_turning_it_off_restores_the_plain_path() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let (ids, targets) = batch_data(&c, 4, 5, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    let loss_of = |qat: bool| {
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 62), None).unwrap();
        tr.set_qat(qat).unwrap();
        let first = tr.step(&ids, &targets, 4, &sc).unwrap().loss;
        let second = tr.step(&ids, &targets, 4, &sc).unwrap().loss;
        (first, second)
    };
    let (plain, qat) = (loss_of(false), loss_of(true));
    assert_ne!(
        plain.0, qat.0,
        "the snapped forward pass sees different weights"
    );
    assert_ne!(plain.1, qat.1);
    // On then off: the next steps are exactly the plain ones.
    let mut toggled = GpuTrainer::new(&g, &Gpt2::init(c, 62), None).unwrap();
    toggled.set_qat(true).unwrap();
    toggled.set_qat(false).unwrap();
    assert!(!toggled.qat_enabled());
    let a = toggled.step(&ids, &targets, 4, &sc).unwrap().loss;
    assert_eq!(a, plain.0);
}

#[test]
fn qat_that_does_not_fit_the_budget_is_refused_before_allocating() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let w = Gpt2::init(c, 63);
    let roomy = static_plan(&c).total() + 1000;
    let mut tr = GpuTrainer::new(
        &g,
        &w,
        Some(Budget {
            bytes: roomy,
            source: Source::Override,
        }),
    )
    .unwrap();
    let before = g.buffers_created();
    let e = tr.set_qat(true).unwrap_err().to_string();
    assert!(e.contains("does not fit"), "{e}");
    assert!(!tr.qat_enabled());
    assert_eq!(g.buffers_created(), before);
    // Enabling twice is a no-op the second time.
    let mut ok = GpuTrainer::new(&g, &w, None).unwrap();
    ok.set_qat(true).unwrap();
    let made = g.buffers_created();
    ok.set_qat(true).unwrap();
    assert_eq!(g.buffers_created(), made);
}

#[test]
fn qat_micro_batches_match_the_whole_batch() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.5,
    };
    let run = |micro: usize| {
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 64), None).unwrap();
        tr.set_qat(true).unwrap();
        for s in 0..3 {
            let (ids, tg) = batch_data(&c, 4, 4, s);
            tr.step_with_micro(&ids, &tg, 4, micro, &sc).unwrap();
        }
        tr.weights_host().unwrap()
    };
    let (whole, split) = (run(4), run(2));
    let e = max_err_models(&whole, &split);
    eprintln!("QAT micro-batching: weights differ by {e:.2e}");
    assert!(e < 1e-4, "{e}");
}
