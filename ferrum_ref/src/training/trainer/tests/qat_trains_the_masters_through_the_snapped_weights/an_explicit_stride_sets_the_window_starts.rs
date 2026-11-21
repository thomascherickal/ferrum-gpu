use super::*;

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
