use super::*;

#[test]
fn gqa_attention_bwd_matches_cpu_for_every_group_ratio() {
    let Some(g) = open_for_test() else { return };
    for shape in crate::ops::llama::tests::GQA_SHAPES {
        let (batch, t, heads, kv, hd) = shape;
        let (rows, qd, kvd) = (batch * t, heads * hd, kv * hd);
        let (q, k, v) = (
            spread(rows * qd, 1, 2.0),
            spread(rows * kvd, 2, 2.0),
            rand_vec(rows * kvd, 3),
        );
        let dout = rand_vec(rows * qd, 4);
        let (_, probs) = cpu::gqa_fwd(&q, &k, &v, shape);
        let (wq, wk, wv) = cpu::gqa_bwd((&q, &k, &v), &probs, &dout, shape);
        let up = |x: &[f32]| g.upload(x).unwrap();
        let (gq, gk, gv, gp, gd) = (up(&q), up(&k), up(&v), up(&probs), up(&dout));
        let (dq, dk, dv) = (
            g.zeros(rows * qd).unwrap(),
            g.zeros(rows * kvd).unwrap(),
            g.zeros(rows * kvd).unwrap(),
        );
        let (dot, ds) = (
            g.zeros(batch * heads * t).unwrap(),
            g.zeros(batch * heads * t * t).unwrap(),
        );
        attention_gqa_bwd(
            &g,
            (&gq, &gk, &gv, &gp, &gd),
            (&dq, &dk, &dv),
            (&dot, &ds),
            shape,
        )
        .unwrap();
        let tag = format!("{shape:?}");
        show(
            "attention_gqa_bwd dq",
            &tag,
            max_rel_err(&g.read(&dq).unwrap(), &wq),
        );
        show(
            "attention_gqa_bwd dk",
            &tag,
            max_rel_err(&g.read(&dk).unwrap(), &wk),
        );
        show(
            "attention_gqa_bwd dv",
            &tag,
            max_rel_err(&g.read(&dv).unwrap(), &wv),
        );
    }
}
