use super::*;

#[test]
fn embedding_grad_with_duplicate_tokens_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    let (vocab, c, t, ctx) = (9usize, 6usize, 4usize, 6usize);
    // Two sequences with many repeated tokens, and token 8 never used.
    let ids = vec![2usize, 2, 5, 2, 0, 5, 5, 2];
    let rows = ids.len();
    let dx = rand_vec(rows * c, 16);
    let mut want_tok = vec![0.0f32; vocab * c];
    let mut want_pos = vec![0.0f32; ctx * c];
    for (r, &id) in ids.iter().enumerate() {
        for i in 0..c {
            want_tok[id * c + i] += dx[r * c + i];
            want_pos[(r % t) * c + i] += dx[r * c + i];
        }
    }
    let (gdx, gt, gp) = (
        g.upload(&dx).unwrap(),
        g.zeros(vocab * c).unwrap(),
        g.zeros(ctx * c).unwrap(),
    );
    embed_bwd(&g, &gdx, &ids, (&gt, &gp), (vocab, c, t, ctx)).unwrap();
    let (got_tok, got_pos) = (g.read(&gt).unwrap(), g.read(&gp).unwrap());
    // Same ascending row order as the CPU loop, so the sums are bit-identical.
    assert_eq!(got_tok, want_tok);
    assert_eq!(got_pos, want_pos);
    assert!(got_tok[8 * c..].iter().all(|&v| v == 0.0));
    assert!(
        got_pos[t * c..].iter().all(|&v| v == 0.0),
        "unused positions get no gradient"
    );
}

#[test]
fn shape_and_range_errors() {
    let Some(g) = open_for_test() else { return };
    let (a, b, c) = (
        g.zeros(4).unwrap(),
        g.zeros(4).unwrap(),
        g.zeros(3).unwrap(),
    );
    assert!(matmul_ex(&g, (&a, &b, &c), (2, 2, 2), (false, false)).is_err());
    assert!(colsum(&g, &a, &c, (2, 3)).is_err());
    assert!(gelu_bwd(&g, &a, &c, &b).is_err());
    let dx = g.zeros(4).unwrap();
    assert!(embed_bwd(&g, &dx, &[1, 9], (&a, &b), (2, 2, 2, 2)).is_err());
    assert!(
        embed_bwd(&g, &dx, &[0, 1], (&a, &b), (2, 2, 3, 2)).is_err(),
        "t above ctx"
    );
}
