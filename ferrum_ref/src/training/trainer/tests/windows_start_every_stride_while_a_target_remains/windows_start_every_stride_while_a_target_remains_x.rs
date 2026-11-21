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
