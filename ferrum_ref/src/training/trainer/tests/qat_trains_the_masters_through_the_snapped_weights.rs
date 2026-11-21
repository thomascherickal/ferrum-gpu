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

#[test]
fn an_explicit_stride_sets_the_window_starts() {
    let toks: Vec<usize> = (0..21).collect();
    let mut m = Mock::default();
    let c = TrainConfig {
        context: 4,
        batch: 100,
        stride: 2,
        epochs: 1,
        ..TrainConfig::default()
    };
    train(&mut m, &toks, &c, None, None, &mut NoHooks).unwrap();
    // one step holding every window at stride 2: starts 0,2,...,16 (a target must remain)
    assert_eq!(m.calls.len(), 1);
    assert_eq!(m.calls[0].2, window_starts(21, 4, 2).len());
    assert!(
        m.calls[0].2 > window_starts(21, 4, 4).len(),
        "overlapping windows are more numerous"
    );
}

#[test]
fn an_engine_failure_ends_the_run_with_its_message() {
    let c = TrainConfig {
        context: 4,
        batch: 2,
        epochs: 1,
        ..TrainConfig::default()
    };
    let e = train(&mut Failing, &corpus(100), &c, None, None, &mut NoHooks).unwrap_err();
    assert!(e.contains("device exploded"), "{e}");
    assert_eq!(Failing.steps(), 0);
    assert!(Failing.snapshot().is_err());
    assert!(Failing
        .restore(&Mock::default().snapshot().unwrap())
        .is_ok());
}

#[test]
fn a_lazy_source_trains_exactly_like_the_same_tokens_in_memory() {
    let toks: Vec<usize> = (0..60).map(|i| (i * 7) % 50).collect();
    let c = TrainConfig {
        context: 6,
        batch: 3,
        epochs: 2,
        ..TrainConfig::default()
    };
    let (mut a, mut b) = (Mock::default(), Mock::default());
    let ra = train(&mut a, &toks, &c, None, None, &mut NoHooks).unwrap();
    let rb = train(&mut b, &Lazy(60), &c, None, None, &mut NoHooks).unwrap();
    assert_eq!(ra, rb);
    assert_eq!(a.calls, b.calls, "the same batches in the same order");
    // a slice and a boxed trait object work too
    let mut s = Mock::default();
    train(&mut s, &toks[..], &c, None, None, &mut NoHooks).unwrap();
    let dynamic: &dyn TokenSource = &Lazy(60);
    train(&mut Mock::default(), dynamic, &c, None, None, &mut NoHooks).unwrap();
    assert_eq!(s.calls, a.calls);
}
