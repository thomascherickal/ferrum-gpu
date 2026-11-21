use super::*;

#[test]
fn cross_entropy_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for (rows, vocab) in [(1usize, 2usize), (3, 5), (16, 1024), (33, 7)] {
        let logits = spread(rows * vocab, 9, 8.0);
        let targets: Vec<usize> = (0..rows).map(|i| (i * 7 + 1) % vocab).collect();
        let (gl, gt) = (
            g.upload(&logits).unwrap(),
            g.upload_indices(&targets, vocab).unwrap(),
        );
        let (gg, gr, gloss) = (
            g.zeros(rows * vocab).unwrap(),
            g.zeros(rows).unwrap(),
            g.zeros(1).unwrap(),
        );
        cross_entropy(&g, &gl, &gt, (&gg, &gr, &gloss), (rows, vocab)).unwrap();
        let (want_loss, want_grad) = nn::cross_entropy(&logits, &targets, vocab);
        show(
            "ce loss",
            &format!("{rows}x{vocab}"),
            max_rel_err(&g.read(&gloss).unwrap(), &[want_loss]),
            REDUCING,
        );
        show(
            "ce grad",
            &format!("{rows}x{vocab}"),
            max_rel_err(&g.read(&gg).unwrap(), &want_grad),
            REDUCING,
        );
    }
}

#[test]
fn score_matches_cpu_including_ties() {
    let Some(g) = open_for_test() else { return };
    let (rows, vocab) = (5usize, 6usize);
    let mut logits = spread(rows * vocab, 11, 4.0);
    logits[vocab..2 * vocab].fill(1.5); // a fully tied row: first index is the argmax
    let targets = vec![0usize, 0, 3, 5, 1];
    let (gl, gt, go) = (
        g.upload(&logits).unwrap(),
        g.upload_indices(&targets, vocab).unwrap(),
        g.zeros(2 * rows).unwrap(),
    );
    score(&g, &gl, &gt, &go, (rows, vocab)).unwrap();
    let got = g.read(&go).unwrap();
    let want = nn::score(&logits, &targets, vocab);
    for (i, (lp, hit)) in want.iter().enumerate() {
        let g0 = got[2 * i];
        assert!(
            (g0 - lp).abs() < 1e-4 * lp.abs().max(1.0),
            "row {i}: {g0} vs {lp}"
        );
        assert_eq!(got[2 * i + 1] == 1.0, *hit, "row {i} argmax flag");
    }
    assert_eq!(got[3], 1.0, "tied row with target 0 counts as argmax");
}

#[test]
fn embed_matches_cpu_and_rejects_bad_ids() {
    let Some(g) = open_for_test() else { return };
    let (vocab, c, t) = (11usize, 6usize, 4usize);
    let (tok, pos) = (rand_vec(vocab * c, 12), rand_vec(t * c, 13));
    let ids = vec![0usize, 10, 3, 3, 7, 1, 2]; // 7 rows, two sequences' worth of positions
    let (gt, gp, gi) = (
        g.upload(&tok).unwrap(),
        g.upload(&pos).unwrap(),
        g.upload_indices(&ids, vocab).unwrap(),
    );
    let go = g.zeros(ids.len() * c).unwrap();
    embed(&g, (&gt, &gp, &gi), &go, (ids.len(), c, t)).unwrap();
    show(
        "embed",
        "7x6",
        max_rel_err(&g.read(&go).unwrap(), &nn::embed(&tok, &pos, &ids, t, c)),
        EXACT,
    );
    let e = g
        .upload_indices(&[3, 11], vocab)
        .expect_err("id 11 is out of range");
    assert!(e.to_string().contains("11"));
}

#[test]
fn multi_command_ops_report_a_fault_at_each_of_their_commands() {
    let Some(g) = open_for_test() else { return };
    // attention: scores, softmax, P*V
    let (batch, t, heads, hd) = (1usize, 3usize, 2usize, 2usize);
    let n = batch * t * heads * hd;
    let (q, k, v) = (
        g.upload(&rand_vec(n, 1)).unwrap(),
        g.upload(&rand_vec(n, 2)).unwrap(),
        g.upload(&rand_vec(n, 3)).unwrap(),
    );
    let (probs, out) = (g.zeros(batch * heads * t * t).unwrap(), g.zeros(n).unwrap());
    assert_eq!(
        g.sweep_faults(|| attention(&g, (&q, &k, &v), (&probs, &out), (batch, t, heads, hd))),
        3
    );
    // decode attention: probs, P*V
    let (kc, vc, pr, o) = (
        g.zeros(8 * 4).unwrap(),
        g.zeros(8 * 4).unwrap(),
        g.zeros(2 * 8).unwrap(),
        g.zeros(4).unwrap(),
    );
    let q1 = g.upload(&rand_vec(4, 4)).unwrap();
    assert_eq!(
        g.sweep_faults(|| attention_decode(&g, (&q1, &kc, &vc), (&pr, &o), (2, 2, 3, 8))),
        2
    );
    // cross-entropy: rows, mean
    let (logits, grad, rl, loss) = (
        g.upload(&rand_vec(6, 5)).unwrap(),
        g.zeros(6).unwrap(),
        g.zeros(2).unwrap(),
        g.zeros(1).unwrap(),
    );
    let tg = g.upload_indices(&[1, 2], 3).unwrap();
    assert_eq!(
        g.sweep_faults(|| cross_entropy(&g, &logits, &tg, (&grad, &rl, &loss), (2, 3))),
        2
    );
    // single-command ops fail on their only command
    let (a, b, c) = (
        g.zeros(4).unwrap(),
        g.zeros(4).unwrap(),
        g.zeros(4).unwrap(),
    );
    assert_eq!(g.sweep_faults(|| add(&g, &a, &b, &c)), 1);
    assert_eq!(g.sweep_faults(|| gelu(&g, &a, &c)), 1);
}

#[test]
fn cross_entropy_with_no_rows_is_an_error_not_a_division_by_zero() {
    let Some(g) = open_for_test() else { return };
    let (logits, grad, row_loss) = (
        g.zeros(0).unwrap(),
        g.zeros(0).unwrap(),
        g.zeros(0).unwrap(),
    );
    let targets = g.upload_indices(&[], 4).unwrap();
    let loss = g.zeros(1).unwrap();
    let e = cross_entropy(&g, &logits, &targets, (&grad, &row_loss, &loss), (0, 4)).unwrap_err();
    assert!(e.to_string().contains("zero rows"), "{e}");
}
