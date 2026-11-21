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
