use super::*;

#[test]
fn presets_are_valid_models_and_unknown_names_are_refused() {
    for name in ["tiny", "small", "medium"] {
        let cfg = preset(name).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.vocab, 256);
        let m = fresh_model(name, 1).unwrap();
        assert!(!m.tied && m.num_params() > 0);
    }
    assert!(preset("tiny").unwrap().ctx < preset("medium").unwrap().ctx);
    assert!(preset("huge").is_none());
    assert!(fresh_model("huge", 1)
        .unwrap_err()
        .contains("unknown preset"));
    assert_eq!(
        fresh_model("tiny", 5).unwrap(),
        fresh_model("tiny", 5).unwrap(),
        "seeded"
    );
    assert_ne!(
        fresh_model("tiny", 5).unwrap(),
        fresh_model("tiny", 6).unwrap()
    );
}

#[test]
fn batches_are_shifted_windows_chosen_by_the_seed() {
    let text = pattern(200);
    let (ids, targets) = make_batch(&text, 10, 3, &mut Rng::new(1));
    assert_eq!((ids.len(), targets.len()), (30, 30));
    for w in 0..3 {
        // the target of each position is the next byte of the same window
        assert_eq!(
            &ids[w * 10 + 1..(w + 1) * 10],
            &targets[w * 10..(w + 1) * 10 - 1]
        );
    }
    assert_eq!(
        make_batch(&text, 10, 3, &mut Rng::new(1)),
        (ids.clone(), targets)
    );
    assert_ne!(make_batch(&text, 10, 3, &mut Rng::new(2)).0, ids);
    // the last window may end exactly at the final byte
    let (i, t) = make_batch(b"abcd", 3, 50, &mut Rng::new(9));
    assert!(i.iter().chain(&t).all(|&x| (97..=100).contains(&x)));
}

#[test]
fn train_specs_are_validated_with_the_reason() {
    let cfg = preset("tiny").unwrap();
    assert!(spec(5).validate(&cfg).is_ok());
    let bad = |f: &dyn Fn(&mut TrainSpec)| {
        let mut s = spec(5);
        f(&mut s);
        s.validate(&cfg).unwrap_err()
    };
    assert!(bad(&|s| s.steps = 0).contains("steps"));
    assert!(bad(&|s| s.steps = 200_000).contains("steps"));
    assert!(bad(&|s| s.batch = 0).contains("batch"));
    assert!(bad(&|s| s.batch = 1000).contains("batch"));
    assert!(bad(&|s| s.context = 0).contains("context"));
    assert!(bad(&|s| s.context = 65).contains("context"));
    assert!(bad(&|s| s.text = vec![1; 24]).contains("needs more"));
    assert!(bad(&|s| s.lr = 0.0).contains("lr"));
    assert!(bad(&|s| s.lr = f32::NAN).contains("lr"));
    assert!(bad(&|s| s.lr = 5.0).contains("lr"));
}

#[test]
fn the_loop_reports_progress_follows_the_schedule_and_can_be_stopped() {
    let progress = shared();
    let mut lrs = Vec::new();
    run_training(&spec(40), &progress, &mut |ids, targets, batch, cfg| {
        assert_eq!((ids.len(), targets.len(), batch), (96, 96, 4));
        lrs.push(cfg.adam.lr);
        assert_eq!(cfg.clip, 1.0);
        Ok(StepStats {
            loss: 2.0,
            grad_norm: 1.0,
        })
    })
    .unwrap();
    let p = progress.lock().unwrap().clone();
    assert_eq!((p.step, p.total, p.losses.len()), (40, 40, 40));
    assert!(
        lrs[0] < lrs[1] && lrs[1] < lrs[3] && (lrs[3] - 0.01).abs() < 1e-6,
        "warmup (4 steps) rises to the peak: {lrs:?}"
    );
    assert!(
        lrs[39] < lrs[10] && lrs[10] < lrs[3],
        "cosine decay falls: {lrs:?}"
    );
    // a stop request ends the run cleanly after the current step
    let progress = shared();
    let flag = progress.clone();
    run_training(&spec(50), &progress, &mut |_, _, _, _| {
        flag.lock().unwrap().stop = true;
        Ok(StepStats {
            loss: 1.0,
            grad_norm: 1.0,
        })
    })
    .unwrap();
    assert_eq!(progress.lock().unwrap().step, 1);
    // a cancelled request (the thread's flag) ends the run before the first step
    let progress = shared();
    {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let _scope = crate::cancel::install(flag);
        run_training(&spec(50), &progress, &mut |_, _, _, _| {
            panic!("no step runs")
        })
        .unwrap();
    }
    assert_eq!(progress.lock().unwrap().step, 0);
    // a diverged loss and an engine error both end the run with a message
    let e = run_training(&spec(5), &shared(), &mut |_, _, _, _| {
        Ok(StepStats {
            loss: f32::NAN,
            grad_norm: 0.0,
        })
    })
    .unwrap_err();
    assert!(e.contains("lower the learning rate"), "{e}");
    let e = run_training(&spec(5), &shared(), &mut |_, _, _, _| {
        Err("device lost".into())
    })
    .unwrap_err();
    assert_eq!(e, "device lost");
}
