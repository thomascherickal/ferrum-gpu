use super::*;

#[test]
fn cross_entropy_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for (rows, vocab) in [(1usize, 2usize), (3, 5), (16, 1024), (33, 7)] {
        let logits = spread(rows * vocab, 9, 8.0);
        let targets: Vec<usize> = (0..rows).map(|i| (i * 7 + 1) % vocab).collect();
        let (grad, loss) = gpu_cross_entropy(&g, &logits, &targets, vocab);
        let (want_loss, want_grad) = nn::cross_entropy(&logits, &targets, vocab);
        show(
            "ce loss",
            &format!("{rows}x{vocab}"),
            max_rel_err(&loss, &[want_loss]),
            REDUCING,
        );
        show(
            "ce grad",
            &format!("{rows}x{vocab}"),
            max_rel_err(&grad, &want_grad),
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
