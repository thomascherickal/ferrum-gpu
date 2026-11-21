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
