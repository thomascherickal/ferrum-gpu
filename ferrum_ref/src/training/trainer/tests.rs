use super::*;
use crate::gpt2::{Config, Gpt2};

fn cfg() -> Config {
    Config {
        vocab: 11,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 16,
    }
}

/// A tiny learnable corpus: a repeating pattern.
fn corpus(n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| [3usize, 7, 1, 9, 4, 4, 2, 8][i % 8])
        .collect()
}

/// Records what the loop asks of an engine, without doing any math.
#[derive(Default)]
struct Mock {
    calls: Vec<(Vec<usize>, Vec<usize>, usize, f32)>,
    done: u64,
}
impl Engine for Mock {
    fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        c: &StepConfig,
    ) -> Result<StepStats, String> {
        self.calls
            .push((ids.to_vec(), targets.to_vec(), batch, c.adam.lr));
        self.done += 1;
        Ok(StepStats {
            loss: self.done as f32,
            grad_norm: 0.0,
        })
    }
    fn steps(&self) -> u64 {
        self.done
    }
    fn snapshot(&self) -> Result<Snapshot, String> {
        let w = Gpt2::random(cfg(), 1);
        Ok(Snapshot {
            weights: w.clone(),
            m: w.zeros_like(),
            v: w.zeros_like(),
            steps: self.done,
        })
    }
    fn restore(&mut self, s: &Snapshot) -> Result<(), String> {
        self.done = s.steps;
        Ok(())
    }
}

#[test]
fn windows_start_every_stride_while_a_target_remains() {
    assert_eq!(window_starts(10, 4, 0), vec![0, 4]); // 0..4+1 and 4..8+1; a window at 8 needs 13 tokens
    assert_eq!(window_starts(9, 4, 0), vec![0, 4]);
    assert_eq!(window_starts(8, 4, 0), vec![0]);
    assert_eq!(window_starts(6, 4, 1), vec![0, 1]);
    assert_eq!(
        window_starts(5, 4, 1),
        vec![0],
        "5 tokens hold exactly one 4+1 window"
    );
    assert_eq!(window_starts(4, 4, 0), Vec::<usize>::new());
    assert_eq!(window_starts(100, 0, 0), Vec::<usize>::new());
    assert_eq!(window_starts(0, 4, 0), Vec::<usize>::new());
}

#[test]
fn every_epoch_visits_every_window_once_in_a_seeded_order() {
    let o0 = epoch_order(20, 5, 0);
    let o1 = epoch_order(20, 5, 1);
    assert_eq!(o0, epoch_order(20, 5, 0), "deterministic");
    assert_ne!(o0, o1, "a fresh shuffle each epoch");
    assert_ne!(o0, epoch_order(20, 6, 0), "depends on the seed");
    for o in [o0, o1] {
        let mut s = o.clone();
        s.sort();
        assert_eq!(s, (0..20).collect::<Vec<_>>());
    }
    assert_eq!(steps_per_epoch(20, 6), 4);
    assert_eq!(steps_per_epoch(20, 5), 4);
    assert_eq!(steps_per_epoch(1, 16), 1);
}

#[test]
fn the_loop_feeds_shifted_windows_and_a_short_last_batch() {
    let toks: Vec<usize> = (0..21).collect(); // context 4: windows at 0,4,8,12,16 (needs 5 tokens: 16..21 ok)
    let mut m = Mock::default();
    let c = TrainConfig {
        context: 4,
        batch: 2,
        epochs: 1,
        ..TrainConfig::default()
    };
    let r = train(&mut m, &toks, &c, None, None, &mut NoHooks).unwrap();
    assert_eq!(r.steps_run, 3, "5 windows at batch 2 = 3 steps");
    let batches: Vec<usize> = m.calls.iter().map(|c| c.2).collect();
    assert_eq!(batches, vec![2, 2, 1], "the last batch is short");
    let order = epoch_order(5, c.seed, 0);
    for (i, call) in m.calls.iter().enumerate() {
        let (ids, tg) = (&call.0, &call.1);
        for (b, &w) in order[i * 2..(i * 2 + call.2)].iter().enumerate() {
            let s = w * 4;
            assert_eq!(&ids[b * 4..(b + 1) * 4], &toks[s..s + 4]);
            assert_eq!(
                &tg[b * 4..(b + 1) * 4],
                &toks[s + 1..s + 5],
                "targets are the next tokens"
            );
        }
    }
}

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

#[derive(Default)]
struct Rec {
    steps: Vec<u64>,
    checkpoints: Vec<(u64, u64, u64)>,
    stop_at: Option<u64>,
}
impl Hooks for Rec {
    fn on_step(&mut self, step: u64, _s: &StepStats) -> bool {
        self.steps.push(step);
        self.stop_at != Some(step)
    }
    fn on_checkpoint(&mut self, cp: &Checkpoint) -> Result<(), String> {
        self.checkpoints
            .push((cp.snapshot.steps, cp.epoch, cp.step_in_epoch));
        Ok(())
    }
}

#[test]
fn checkpoints_follow_the_cadence_and_always_come_at_the_end() {
    let toks = corpus(100);
    let mut m = Mock::default();
    let c = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 7,
        checkpoint_every: 3,
        ..TrainConfig::default()
    };
    let mut h = Rec::default();
    train(&mut m, &toks, &c, None, Some("sha256:abc"), &mut h).unwrap();
    assert_eq!(h.steps, (1..=7).collect::<Vec<_>>());
    // 12 windows / batch 4 = 3 steps per epoch: step 3 -> (epoch 1, 0), step 6 -> (epoch 2, 0), end at 7.
    assert_eq!(h.checkpoints, vec![(3, 1, 0), (6, 2, 0), (7, 2, 1)]);
    // Without a cadence there is still one at the end.
    let mut m = Mock::default();
    let mut h = Rec::default();
    train(
        &mut m,
        &toks,
        &TrainConfig {
            checkpoint_every: 0,
            ..c.clone()
        },
        None,
        None,
        &mut h,
    )
    .unwrap();
    assert_eq!(h.checkpoints, vec![(7, 2, 1)]);
}

#[test]
fn a_stop_request_ends_the_run_with_a_final_checkpoint() {
    let toks = corpus(100);
    let mut m = Mock::default();
    let c = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 50,
        ..TrainConfig::default()
    };
    let mut h = Rec {
        stop_at: Some(4),
        ..Rec::default()
    };
    let r = train(&mut m, &toks, &c, None, None, &mut h).unwrap();
    assert!(r.stopped);
    assert_eq!((r.steps_run, r.final_step), (4, 4));
    assert_eq!(h.checkpoints, vec![(4, 1, 1)]);
}

#[test]
fn a_failing_checkpoint_hook_fails_the_run() {
    struct Bad;
    impl Hooks for Bad {
        fn on_checkpoint(&mut self, _: &Checkpoint) -> Result<(), String> {
            Err("disk full".into())
        }
    }
    let mut m = Mock::default();
    let c = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 2,
        ..TrainConfig::default()
    };
    assert_eq!(
        train(&mut m, &corpus(100), &c, None, None, &mut Bad).unwrap_err(),
        "disk full"
    );
}

#[test]
fn invalid_configs_and_corpora_are_refused() {
    let m = Mock::default();
    let ok = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 1,
        ..TrainConfig::default()
    };
    let e = |c: &TrainConfig, t: &[usize]| {
        train(&mut Mock::default(), t, c, None, None, &mut NoHooks).unwrap_err()
    };
    assert!(e(
        &TrainConfig {
            context: 0,
            ..ok.clone()
        },
        &corpus(50)
    )
    .contains("positive"));
    assert!(e(
        &TrainConfig {
            batch: 0,
            ..ok.clone()
        },
        &corpus(50)
    )
    .contains("positive"));
    assert!(e(&ok, &corpus(8)).contains("too few"));
    assert!(e(&ok, &[]).contains("too few"));
    assert_eq!(m.calls.len(), 0);
}

#[test]
fn resume_refuses_a_changed_run_shape() {
    let toks = corpus(100);
    let c = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 3,
        ..TrainConfig::default()
    };
    let mut m = Mock::default();
    let mut h = Rec::default();
    train(&mut m, &toks, &c, None, None, &mut h).unwrap();
    let shape = RunShape {
        context: 8,
        batch: 4,
        stride: 8,
        windows: 12,
        seed: c.seed,
    };
    // Same shape resumes; any change is refused.
    let mut again = Mock {
        done: 3,
        ..Mock::default()
    };
    assert!(train(
        &mut again,
        &toks,
        &TrainConfig {
            max_steps: 5,
            ..c.clone()
        },
        Some(&shape),
        None,
        &mut NoHooks
    )
    .is_ok());
    for bad in [
        TrainConfig {
            batch: 3,
            ..c.clone()
        },
        TrainConfig {
            seed: 9,
            ..c.clone()
        },
        TrainConfig {
            context: 4,
            ..c.clone()
        },
    ] {
        let e = train(
            &mut Mock::default(),
            &toks,
            &bad,
            Some(&shape),
            None,
            &mut NoHooks,
        )
        .unwrap_err();
        assert!(e.contains("cannot resume"), "{e}");
    }
    assert!(train(
        &mut Mock::default(),
        &corpus(60),
        &c,
        Some(&shape),
        None,
        &mut NoHooks
    )
    .unwrap_err()
    .contains("cannot resume"));
}

#[test]
fn the_cpu_engine_learns_a_periodic_corpus() {
    let toks = corpus(400);
    let mut e = CpuTrainer::new(Gpt2::init(cfg(), 3), 1);
    let c = TrainConfig {
        context: 8,
        batch: 8,
        epochs: 30,
        lr: 0.02,
        clip: 1.0,
        ..TrainConfig::default()
    };
    let r = train(&mut e, &toks, &c, None, None, &mut NoHooks).unwrap();
    let (first, last) = (r.losses[0], *r.losses.last().unwrap());
    assert!(first > 2.0, "starts near ln(11) = 2.4, got {first}");
    assert!(last < first * 0.25, "learns the pattern: {first} -> {last}");
}

fn tied(mut w: Gpt2) -> Gpt2 {
    w.tie_head();
    w
}

#[test]
fn the_folded_gradient_of_a_tied_model_matches_finite_differences() {
    use crate::grad::{loss_and_grads, transpose};
    let c = cfg();
    let toks = corpus(17);
    let (ids, targets) = (&toks[..16], &toks[1..17]);
    let base = tied(Gpt2::init(c, 5));
    assert_eq!(base.head, transpose(&base.tok, c.vocab, c.dim));
    let (_, mut g) = loss_and_grads(&base, ids, targets, 2, 1).unwrap();
    g.fold_head_grad();
    assert!(
        g.head.iter().all(|&x| x == 0.0),
        "the head's own gradient is cleared"
    );
    let loss_at = |tok: &[f32]| {
        let mut m = base.clone();
        m.tok = tok.to_vec();
        m.tie_head();
        loss_and_grads(&m, ids, targets, 2, 1).unwrap().0
    };
    let eps = 5e-3f32;
    // entries of tokens that occur as input (embedding path) and as targets (head path)
    for &(t, k) in &[(3usize, 0usize), (7, 3), (1, 5), (9, 2), (4, 7), (0, 1)] {
        let i = t * c.dim + k;
        let (mut hi, mut lo) = (base.tok.clone(), base.tok.clone());
        hi[i] += eps;
        lo[i] -= eps;
        let numeric = (loss_at(&hi) - loss_at(&lo)) / (2.0 * eps);
        let analytic = g.tok[i];
        assert!(
            (numeric - analytic).abs() <= 3e-3 + 0.05 * analytic.abs(),
            "tok[{t}][{k}]: numeric {numeric} vs folded {analytic}"
        );
    }
}

#[test]
fn a_tied_trainer_keeps_the_head_exactly_the_embedding_transposed_and_still_learns() {
    use crate::grad::transpose;
    let toks = corpus(400);
    let c = cfg();
    let tc = TrainConfig {
        context: 8,
        batch: 8,
        epochs: 30,
        lr: 0.02,
        clip: 1.0,
        weight_decay: 0.01,
        ..TrainConfig::default()
    };
    for qat in [false, true] {
        let mut e = CpuTrainer::new(Gpt2::init(c, 3), 1)
            .with_tie(true)
            .with_qat(qat);
        let r = train(&mut e, &toks, &tc, None, None, &mut NoHooks).unwrap();
        let w = e.weights();
        assert_eq!(
            w.head,
            transpose(&w.tok, c.vocab, c.dim),
            "qat={qat}: still tied"
        );
        let (first, last) = (r.losses[0], *r.losses.last().unwrap());
        assert!(
            last < first * 0.35,
            "qat={qat}: learns the pattern: {first} -> {last}"
        );
    }
    // an untied run on the same data does not keep that relation
    let mut u = CpuTrainer::new(Gpt2::init(c, 3), 1);
    train(&mut u, &toks, &tc, None, None, &mut NoHooks).unwrap();
    assert_ne!(
        u.weights().head,
        transpose(&u.weights().tok, c.vocab, c.dim)
    );
}

#[test]
fn restoring_an_untied_snapshot_into_a_tied_trainer_ties_it() {
    use crate::grad::transpose;
    let c = cfg();
    let untied = CpuTrainer::new(Gpt2::init(c, 9), 1);
    let snap = untied.snapshot().unwrap();
    let mut t = CpuTrainer::new(Gpt2::init(c, 1), 1).with_tie(true);
    t.restore(&snap).unwrap();
    let w = t.weights();
    assert_eq!(
        w.tok, snap.weights.tok,
        "the embedding comes from the snapshot"
    );
    assert_eq!(w.head, transpose(&w.tok, c.vocab, c.dim));
}

#[test]
fn resuming_from_a_snapshot_equals_an_uninterrupted_run() {
    let toks = corpus(300);
    let c = TrainConfig {
        context: 8,
        batch: 6,
        max_steps: 9,
        lr: 0.01,
        weight_decay: 0.01,
        clip: 0.5,
        cosine: true,
        warmup: 2,
        ..TrainConfig::default()
    };
    let mut straight = CpuTrainer::new(Gpt2::init(cfg(), 4), 1);
    train(&mut straight, &toks, &c, None, None, &mut NoHooks).unwrap();
    // Run 4 steps, take the checkpoint the loop offers, and finish in a fresh engine.
    let mut first = CpuTrainer::new(Gpt2::init(cfg(), 4), 1);
    let mut h = Rec {
        stop_at: Some(4),
        ..Rec::default()
    };
    struct Keep(Option<Checkpoint>, Rec);
    impl Hooks for Keep {
        fn on_step(&mut self, s: u64, st: &StepStats) -> bool {
            self.1.on_step(s, st)
        }
        fn on_checkpoint(&mut self, cp: &Checkpoint) -> Result<(), String> {
            self.0 = Some(cp.clone());
            Ok(())
        }
    }
    let mut keep = Keep(None, std::mem::take(&mut h));
    train(&mut first, &toks, &c, None, None, &mut keep).unwrap();
    let cp = keep.0.expect("a checkpoint at the stop");
    assert_eq!(cp.snapshot.steps, 4);
    let mut resumed = CpuTrainer::new(Gpt2::init(cfg(), 99), 1);
    resumed.restore(&cp.snapshot).unwrap();
    let r = train(&mut resumed, &toks, &c, Some(&cp.shape), None, &mut NoHooks).unwrap();
    assert_eq!(r.steps_run, 5);
    assert_eq!(
        resumed.weights(),
        straight.weights(),
        "bit-identical to the uninterrupted run"
    );
    assert_eq!(resumed.snapshot().unwrap(), straight.snapshot().unwrap());
}

#[test]
fn the_mock_engine_restores_its_step_counter() {
    let mut m = Mock::default();
    let snap = m.snapshot().unwrap();
    let mut snap = snap;
    snap.steps = 9;
    m.restore(&snap).unwrap();
    assert_eq!(m.steps(), 9);
    // Resuming a mock at step 9 of a 10-step budget runs exactly one more step.
    let c = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 10,
        ..TrainConfig::default()
    };
    assert_eq!(
        train(&mut m, &corpus(100), &c, None, None, &mut NoHooks)
            .unwrap()
            .steps_run,
        1
    );
}

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

/// An engine whose every step fails.
struct Failing;
impl Engine for Failing {
    fn step(
        &mut self,
        _: &[usize],
        _: &[usize],
        _: usize,
        _: &StepConfig,
    ) -> Result<StepStats, String> {
        Err("device exploded".into())
    }
    fn steps(&self) -> u64 {
        0
    }
    fn snapshot(&self) -> Result<Snapshot, String> {
        Err("no state".into())
    }
    fn restore(&mut self, _: &Snapshot) -> Result<(), String> {
        Ok(())
    }
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

/// A source that computes its tokens on demand, like a file read window by window.
struct Lazy(usize);

impl TokenSource for Lazy {
    fn len(&self) -> usize {
        self.0
    }
    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        if start + len > self.0 {
            return Err("past the end".into());
        }
        Ok((start..start + len).map(|i| (i * 7) % 50).collect())
    }
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

#[test]
fn a_prefix_hides_the_held_out_tail() {
    let toks: Vec<usize> = (0..30).collect();
    let p = Prefix {
        inner: &toks,
        len: 20,
    };
    assert_eq!((TokenSource::len(&p), p.is_empty()), (20, false));
    assert_eq!(p.window(5, 3).unwrap(), vec![5, 6, 7]);
    assert!(p.window(18, 3).unwrap_err().contains("first 20"));
    let c = TrainConfig {
        context: 4,
        batch: 2,
        epochs: 1,
        ..TrainConfig::default()
    };
    let mut m = Mock::default();
    train(&mut m, &p, &c, None, None, &mut NoHooks).unwrap();
    assert!(m
        .calls
        .iter()
        .all(|call| call.0.iter().chain(&call.1).all(|&t| t < 20)));
    assert!(
        Prefix {
            inner: &toks,
            len: 99
        }
        .len()
            == 30,
        "never longer than the source"
    );
    assert!(TokenSource::is_empty(&Vec::<usize>::new()));
    assert!(toks.window(25, 9).unwrap_err().contains("outside"));
}
