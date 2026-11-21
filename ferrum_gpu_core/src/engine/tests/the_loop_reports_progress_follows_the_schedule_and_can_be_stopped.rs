use super::*;

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
