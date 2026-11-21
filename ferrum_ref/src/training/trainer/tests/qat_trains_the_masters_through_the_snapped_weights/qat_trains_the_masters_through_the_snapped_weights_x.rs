use super::*;

#[test]
fn qat_trains_the_masters_through_the_snapped_weights() {
    // A model wide enough for Q8_0 (dim 32). With QAT the step's gradient comes from the
    // snapped copy, so the update differs from plain training, yet both reduce the loss.
    let c = Config {
        vocab: 11,
        ctx: 8,
        dim: 32,
        heads: 2,
        layers: 1,
        hidden: 64,
    };
    let toks = corpus(300);
    let tc = TrainConfig {
        context: 8,
        batch: 6,
        epochs: 12,
        lr: 0.01,
        clip: 1.0,
        ..TrainConfig::default()
    };
    let mut plain = CpuTrainer::new(Gpt2::init(c, 5), 1);
    let mut qat = CpuTrainer::new(Gpt2::init(c, 5), 1).with_qat(true);
    let rp = train(&mut plain, &toks, &tc, None, None, &mut NoHooks).unwrap();
    let rq = train(&mut qat, &toks, &tc, None, None, &mut NoHooks).unwrap();
    assert_ne!(
        rp.losses[3], rq.losses[3],
        "the snapped forward pass changes the trajectory"
    );
    for r in [&rp, &rq] {
        assert!(
            *r.losses.last().unwrap() < r.losses[0] * 0.6,
            "{} -> {}",
            r.losses[0],
            r.losses.last().unwrap()
        );
    }
    // The masters stay full precision: they are not on the Q8_0 grid.
    assert_ne!(qat.weights().fake_quantize(), *qat.weights());
    // A model too narrow to quantise trains identically with or without QAT.
    let narrow = Config {
        dim: 8,
        hidden: 16,
        ..c
    };
    let (mut a, mut b) = (
        CpuTrainer::new(Gpt2::init(narrow, 6), 1),
        CpuTrainer::new(Gpt2::init(narrow, 6), 1).with_qat(true),
    );
    let small = TrainConfig {
        max_steps: 5,
        ..tc.clone()
    };
    train(&mut a, &toks, &small, None, None, &mut NoHooks).unwrap();
    train(&mut b, &toks, &small, None, None, &mut NoHooks).unwrap();
    assert_eq!(a.weights(), b.weights());
}

#[test]
fn cpu_engine_restore_checks_the_shape() {
    let mut e = CpuTrainer::new(Gpt2::init(cfg(), 1), 1);
    let other = Gpt2::init(Config { dim: 12, ..cfg() }, 1);
    let snap = Snapshot {
        weights: other.clone(),
        m: other.zeros_like(),
        v: other.zeros_like(),
        steps: 0,
    };
    assert!(e.restore(&snap).unwrap_err().contains("shape"));
}
