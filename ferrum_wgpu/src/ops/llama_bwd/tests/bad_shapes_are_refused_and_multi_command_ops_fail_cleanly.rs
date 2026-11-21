use super::*;

#[test]
fn bad_shapes_are_refused_and_multi_command_ops_fail_cleanly() {
    let Some(g) = open_for_test() else { return };
    let z = |n: usize| g.zeros(n).unwrap();
    let (a, b) = (z(2 * 4), z(4));
    assert!(rmsnorm_bwd(&g, (&a, &b, &a), (&a, &z(3)), &z(2), (2, 4), 1e-5).is_err());
    assert!(rmsnorm_bwd(&g, (&a, &b, &a), (&a, &b), &z(1), (2, 4), 1e-5).is_err());
    assert!(swiglu_bwd(&g, (&a, &a, &a), (&a, &z(3))).is_err());
    let e = attention_gqa_bwd(
        &g,
        (&a, &a, &a, &a, &a),
        (&a, &a, &a),
        (&a, &a),
        (1, 2, 3, 2, 4),
    );
    assert!(e.unwrap_err().to_string().contains("multiple"));
    // fault sweeps: one command per kernel, so rmsnorm_bwd is 2, swiglu_bwd 1, GQA attention 6
    let (x, gain, dy, dx, dg, st) = (z(8), z(4), z(8), z(8), z(4), z(2));
    assert_eq!(
        g.sweep_faults(|| rmsnorm_bwd(&g, (&x, &gain, &dy), (&dx, &dg), &st, (2, 4), 1e-5)),
        2
    );
    let (gate, up, dact, dgate, dup) = (z(8), z(8), z(8), z(8), z(8));
    assert_eq!(
        g.sweep_faults(|| swiglu_bwd(&g, (&gate, &up, &dact), (&dgate, &dup))),
        1
    );
    let (batch, t, heads, kv, hd) = (1, 3, 4, 2, 2);
    let (q, kk, probs, dout) = (
        z(t * heads * hd),
        z(t * kv * hd),
        z(heads * t * t),
        z(t * heads * hd),
    );
    let (dq, dk, dv) = (z(t * heads * hd), z(t * kv * hd), z(t * kv * hd));
    let (dot, ds) = (z(heads * t), z(heads * t * t));
    let n = g.sweep_faults(|| {
        attention_gqa_bwd(
            &g,
            (&q, &kk, &kk, &probs, &dout),
            (&dq, &dk, &dv),
            (&dot, &ds),
            (batch, t, heads, kv, hd),
        )
    });
    assert_eq!(n, 6);
}
