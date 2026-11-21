use super::*;

#[test]
fn multi_command_backward_ops_report_a_fault_at_each_of_their_commands() {
    let Some(g) = open_for_test() else { return };
    let (m, k, n) = (3usize, 4usize, 5usize);
    let (x, w, dy) = (
        g.zeros(m * k).unwrap(),
        g.zeros(k * n).unwrap(),
        g.zeros(m * n).unwrap(),
    );
    let (dx, dw, db) = (
        g.zeros(m * k).unwrap(),
        g.zeros(k * n).unwrap(),
        g.zeros(n).unwrap(),
    );
    // dx matmul, dw matmul, bias column sum
    assert_eq!(
        g.sweep_faults(|| linear_bwd(&g, (&x, &w, &dy), (&dx, &dw, &db), (m, k, n))),
        3
    );
    // layernorm: input gradient, parameter gradients
    let (rows, cols) = (3usize, 4usize);
    let (lx, lg, ldy) = (
        g.zeros(rows * cols).unwrap(),
        g.zeros(cols).unwrap(),
        g.zeros(rows * cols).unwrap(),
    );
    let (ldx, dg, ds, st) = (
        g.zeros(rows * cols).unwrap(),
        g.zeros(cols).unwrap(),
        g.zeros(cols).unwrap(),
        g.zeros(2 * rows).unwrap(),
    );
    assert_eq!(
        g.sweep_faults(|| layernorm_bwd(&g, (&lx, &lg, &ldy), (&ldx, &dg, &ds), &st, (rows, cols))),
        2
    );
    // attention backward: dP, dot, dS, dV, dQ, dK
    let shape = (1usize, 3usize, 2usize, 2usize);
    let nn_ = shape.0 * shape.1 * shape.2 * shape.3;
    let z = |len: usize| g.zeros(len).unwrap();
    let (q, kk, v, dout, dq, dk, dv) = (z(nn_), z(nn_), z(nn_), z(nn_), z(nn_), z(nn_), z(nn_));
    let (probs, dot, dsb) = (
        z(shape.0 * shape.2 * shape.1 * shape.1),
        z(shape.0 * shape.2 * shape.1),
        z(shape.0 * shape.2 * shape.1 * shape.1),
    );
    assert_eq!(
        g.sweep_faults(|| attention_bwd(
            &g,
            (&q, &kk, &v, &probs, &dout),
            (&dq, &dk, &dv),
            (&dot, &dsb),
            shape
        )),
        6
    );
    // embedding gradient: token rows, position rows
    let (edx, dtok, dpos) = (z(2 * 3), z(4 * 3), z(2 * 3));
    assert_eq!(
        g.sweep_faults(|| embed_bwd(&g, &edx, &[1, 2], (&dtok, &dpos), (4, 3, 2, 2))),
        2
    );
    // gelu backward and colsum are single commands
    let (a, b, c) = (z(4), z(4), z(4));
    assert_eq!(g.sweep_faults(|| gelu_bwd(&g, &a, &b, &c)), 1);
}

#[test]
fn token_csr_is_a_stable_counting_sort() {
    let (start, order) = token_csr(&[2, 0, 2, 2, 1, 0], 4).unwrap();
    assert_eq!(start, vec![0, 2, 3, 6, 6]);
    assert_eq!(
        order,
        vec![1, 5, 4, 0, 2, 3],
        "rows of each token stay in ascending order"
    );
    assert_eq!(token_csr(&[], 3).unwrap(), (vec![0, 0, 0, 0], vec![]));
    assert!(token_csr(&[3], 3)
        .unwrap_err()
        .to_string()
        .contains("out of range"));
}

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
