use super::*;

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
