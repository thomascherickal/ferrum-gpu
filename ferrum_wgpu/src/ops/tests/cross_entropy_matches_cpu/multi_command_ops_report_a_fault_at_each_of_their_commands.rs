use super::*;

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
