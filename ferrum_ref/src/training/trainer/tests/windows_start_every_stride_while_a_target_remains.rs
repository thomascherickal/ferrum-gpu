use super::*;

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
