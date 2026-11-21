use super::*;

#[test]
fn the_cpu_engine_learns_a_periodic_corpus() {
    let toks = corpus(400);
    let mut e = CpuTrainer::new(Gpt2::init(cfg(), 3), 1);
    let c = TrainConfig {
        context: 8,
        batch: 8,
        epochs: 30,
        lr: 0.02,
        clip: 1.0,
        ..TrainConfig::default()
    };
    let r = train(&mut e, &toks, &c, None, None, &mut NoHooks).unwrap();
    let (first, last) = (r.losses[0], *r.losses.last().unwrap());
    assert!(first > 2.0, "starts near ln(11) = 2.4, got {first}");
    assert!(last < first * 0.25, "learns the pattern: {first} -> {last}");
}

#[test]
fn the_folded_gradient_of_a_tied_model_matches_finite_differences() {
    use crate::grad::{loss_and_grads, transpose};
    let c = cfg();
    let toks = corpus(17);
    let (ids, targets) = (&toks[..16], &toks[1..17]);
    let base = tied(Gpt2::init(c, 5));
    assert_eq!(base.head, transpose(&base.tok, c.vocab, c.dim));
    let (_, mut g) = loss_and_grads(&base, ids, targets, 2, 1).unwrap();
    g.fold_head_grad();
    assert!(
        g.head.iter().all(|&x| x == 0.0),
        "the head's own gradient is cleared"
    );
    let loss_at = |tok: &[f32]| {
        let mut m = base.clone();
        m.tok = tok.to_vec();
        m.tie_head();
        loss_and_grads(&m, ids, targets, 2, 1).unwrap().0
    };
    let eps = 5e-3f32;
    // entries of tokens that occur as input (embedding path) and as targets (head path)
    for &(t, k) in &[(3usize, 0usize), (7, 3), (1, 5), (9, 2), (4, 7), (0, 1)] {
        let i = t * c.dim + k;
        let (mut hi, mut lo) = (base.tok.clone(), base.tok.clone());
        hi[i] += eps;
        lo[i] -= eps;
        let numeric = (loss_at(&hi) - loss_at(&lo)) / (2.0 * eps);
        let analytic = g.tok[i];
        assert!(
            (numeric - analytic).abs() <= 3e-3 + 0.05 * analytic.abs(),
            "tok[{t}][{k}]: numeric {numeric} vs folded {analytic}"
        );
    }
}
