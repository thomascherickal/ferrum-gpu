use super::*;

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
