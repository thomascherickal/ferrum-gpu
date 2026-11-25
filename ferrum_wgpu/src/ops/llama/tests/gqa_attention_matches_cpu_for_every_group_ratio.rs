use super::*;

#[test]
fn gqa_attention_matches_cpu_for_every_group_ratio() {
    let Some(g) = open_for_test() else { return };
    for shape in GQA_SHAPES {
        let (batch, t, heads, kv, hd) = shape;
        let (q, k, v) = (
            spread(batch * t * heads * hd, 1, 2.0),
            spread(batch * t * kv * hd, 2, 2.0),
            rand_vec(batch * t * kv * hd, 3),
        );
        let (gq, gk, gv) = (
            g.upload(&q).unwrap(),
            g.upload(&k).unwrap(),
            g.upload(&v).unwrap(),
        );
        let (gp, go) = (
            g.zeros(batch * heads * t * t).unwrap(),
            g.zeros(q.len()).unwrap(),
        );
        attention_gqa(&g, (&gq, &gk, &gv), (&gp, &go), shape).unwrap();
        show(
            "attention_gqa",
            &format!("{shape:?}"),
            max_rel_err(&g.read(&go).unwrap(), &cpu_gqa(&q, &k, &v, shape)),
            REDUCING,
        );
        // probabilities: each causal row sums to one, the future is exactly zero
        let probs = g.read(&gp).unwrap();
        for row in 0..batch * heads * t {
            let (i, p) = (row % t, &probs[row * t..(row + 1) * t]);
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(p[i + 1..].iter().all(|&x| x == 0.0));
        }
    }
}

#[test]
fn gqa_decode_matches_the_last_row_of_full_attention() {
    let Some(g) = open_for_test() else { return };
    // (heads, kv_heads, hd, len, ctx)
    for (heads, kv, hd, len, ctx) in [
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (4, 2, 8, 5, 8),
        (6, 2, 4, 1, 3),
        (4, 4, 3, 7, 7),
        (8, 1, 5, 6, 12),
    ] {
        let (qd, kvd) = (heads * hd, kv * hd);
        let (q, kc, vc) = (
            spread(qd, 4, 2.0),
            spread(ctx * kvd, 5, 2.0),
            rand_vec(ctx * kvd, 6),
        );
        let (gq, gk, gv) = (
            g.upload(&q).unwrap(),
            g.upload(&kc).unwrap(),
            g.upload(&vc).unwrap(),
        );
        let (gp, go) = (g.zeros(heads * ctx).unwrap(), g.zeros(qd).unwrap());
        attention_decode_gqa(&g, (&gq, &gk, &gv), (&gp, &go), (heads, kv, hd, len, ctx)).unwrap();
        let want = cpu::attend_row(&q, &kc, &vc, len, (heads, kv, hd));
        show(
            "attention_decode_gqa",
            &format!("{heads}/{kv} hd={hd} len={len}"),
            max_rel_err(&g.read(&go).unwrap(), &want),
            REDUCING,
        );
    }
}
