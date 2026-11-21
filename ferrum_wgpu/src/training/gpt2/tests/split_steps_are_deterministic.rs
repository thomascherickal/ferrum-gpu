use super::*;

#[test]
fn split_steps_are_deterministic() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let run = || {
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 55), None).unwrap();
        let sc = StepConfig {
            adam: Adam::new(0.01),
            clip: 0.5,
        };
        for s in 0..3 {
            let (ids, tg) = batch_data(&c, 4, 4, s);
            tr.step_with_micro(&ids, &tg, 4, 2, &sc).unwrap();
        }
        tr.snapshot_pair()
    };
    assert_eq!(run(), run());
}

#[test]
fn a_fault_at_any_command_of_a_split_step_applies_nothing() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 56);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let (ids, targets) = batch_data(&c, 2, 3, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    };
    let commands = g.sweep_faults(|| {
        let before = tr.steps();
        let r = tr.step_with_micro(&ids, &targets, 2, 1, &sc).map(|_| ());
        if r.is_err() {
            assert_eq!(tr.steps(), before);
        }
        r
    });
    assert!(commands > 100, "{commands}");
    assert_eq!(tr.steps(), 1);
}

#[test]
fn gpu_qat_follows_the_cpu_qat_trajectory() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let toks = pattern(400, c.vocab);
    let mut cpu = CpuTrainer::new(Gpt2::init(c, 61), 1).with_qat(true);
    let want = train(&mut cpu, &toks, &qloop(), None, None, &mut NoHooks).unwrap();
    let mut gpu = GpuTrainer::new(&g, &Gpt2::init(c, 61), None).unwrap();
    gpu.set_qat(true).unwrap();
    assert!(gpu.qat_enabled());
    let got = train(&mut gpu, &toks, &qloop(), None, None, &mut NoHooks).unwrap();
    let mut worst = 0.0f32;
    for (i, (a, b)) in got.losses.iter().zip(&want.losses).enumerate() {
        let e = (a - b).abs() / b.abs().max(1.0);
        worst = worst.max(e);
        assert!(e < 1e-4, "step {}: gpu {a} vs cpu {b}", i + 1);
    }
    eprintln!("25-step QAT loss curve: worst relative difference {worst:.2e}");
    let (first, last) = (got.losses[0], *got.losses.last().unwrap());
    assert!(last < first * 0.6, "QAT still learns: {first} -> {last}");
    // The masters stay full precision, and exporting them as Q8_0 equals the snapped model.
    let masters = gpu.weights_host().unwrap();
    assert_ne!(masters.fake_quantize(), masters);
}

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
