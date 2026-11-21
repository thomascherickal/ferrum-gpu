use super::*;

#[test]
fn a_tied_trainer_keeps_the_head_exactly_the_embedding_transposed_and_still_learns() {
    use crate::grad::transpose;
    let toks = corpus(400);
    let c = cfg();
    let tc = TrainConfig {
        context: 8,
        batch: 8,
        epochs: 30,
        lr: 0.02,
        clip: 1.0,
        weight_decay: 0.01,
        ..TrainConfig::default()
    };
    for qat in [false, true] {
        let mut e = CpuTrainer::new(Gpt2::init(c, 3), 1)
            .with_tie(true)
            .with_qat(qat);
        let r = train(&mut e, &toks, &tc, None, None, &mut NoHooks).unwrap();
        let w = e.weights();
        assert_eq!(
            w.head,
            transpose(&w.tok, c.vocab, c.dim),
            "qat={qat}: still tied"
        );
        let (first, last) = (r.losses[0], *r.losses.last().unwrap());
        assert!(
            last < first * 0.35,
            "qat={qat}: learns the pattern: {first} -> {last}"
        );
    }
    // an untied run on the same data does not keep that relation
    let mut u = CpuTrainer::new(Gpt2::init(c, 3), 1);
    train(&mut u, &toks, &tc, None, None, &mut NoHooks).unwrap();
    assert_ne!(
        u.weights().head,
        transpose(&u.weights().tok, c.vocab, c.dim)
    );
}

#[test]
fn restoring_an_untied_snapshot_into_a_tied_trainer_ties_it() {
    use crate::grad::transpose;
    let c = cfg();
    let untied = CpuTrainer::new(Gpt2::init(c, 9), 1);
    let snap = untied.snapshot().unwrap();
    let mut t = CpuTrainer::new(Gpt2::init(c, 1), 1).with_tie(true);
    t.restore(&snap).unwrap();
    let w = t.weights();
    assert_eq!(
        w.tok, snap.weights.tok,
        "the embedding comes from the snapshot"
    );
    assert_eq!(w.head, transpose(&w.tok, c.vocab, c.dim));
}
