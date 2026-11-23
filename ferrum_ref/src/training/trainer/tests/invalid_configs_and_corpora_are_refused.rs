use super::*;

#[test]
fn invalid_configs_and_corpora_are_refused() {
    let m = Mock::default();
    let ok = TrainConfig {
        context: 8,
        batch: 4,
        max_steps: 1,
        ..TrainConfig::default()
    };
    let e = |c: &TrainConfig, t: &Vec<usize>| {
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
    assert!(e(&ok, &vec![]).contains("too few"));
    assert_eq!(m.calls.len(), 0);
}
