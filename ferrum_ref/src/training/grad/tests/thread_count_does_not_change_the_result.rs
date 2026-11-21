use super::*;

#[test]
fn thread_count_does_not_change_the_result() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 24);
    let (ids, targets) = data(&cfg, 2, 4);
    let one = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    for t in [2, 5] {
        let other = loss_and_grads(&m, &ids, &targets, 2, t).unwrap();
        assert_eq!(one.0, other.0);
        assert_eq!(one.1, other.1, "bit-identical at {t} threads");
    }
}

#[test]
fn input_validation() {
    let m = Gpt2::random(tiny(), 25);
    let e = |ids: &[usize], tg: &[usize], b: usize| loss_and_grads(&m, ids, tg, b, 1).unwrap_err();
    assert!(e(&[], &[], 1).contains("split"));
    assert!(e(&[1, 2, 3], &[1, 2, 3], 2).contains("split"));
    assert!(e(&[1; 6], &[1; 6], 1).contains("exceeds the context"));
    assert!(e(&[1, 2], &[1], 1).contains("targets"));
    assert!(e(&[7], &[1], 1).contains("out of range"));
    assert!(e(&[1], &[9], 1).contains("out of range"));
}

#[test]
fn loss_matches_the_forward_oracle() {
    let cfg = tiny();
    let m = Gpt2::random(cfg, 26);
    let (ids, targets) = data(&cfg, 1, 5);
    let logits = m.forward_logits(&ids).unwrap();
    let (want, _) = nn::cross_entropy(&logits, &targets, cfg.vocab);
    let (got, _) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
    assert!((got - want).abs() < 1e-5);
}
