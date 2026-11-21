use super::*;

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
