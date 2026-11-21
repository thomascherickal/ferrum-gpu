use super::*;

#[test]
fn gqa_attention_matches_cpu_for_every_group_ratio() {
    let Some(g) = open_for_test() else { return };
    // (batch, t, heads, kv_heads, hd)
    for shape in [
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (1, 4, 4, 4, 3),
        (1, 9, 4, 2, 8),
        (2, 5, 4, 1, 3),
        (1, 17, 6, 2, 8),
        (2, 33, 2, 2, 5),
    ] {
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

#[test]
fn attention_refuses_bad_shapes() {
    let Some(g) = open_for_test() else { return };
    let z = g.zeros(48).unwrap();
    assert!(attention_gqa(&g, (&z, &z, &z), (&z, &z), (1, 2, 3, 2, 4))
        .unwrap_err()
        .to_string()
        .contains("multiple"));
    assert!(attention_gqa(&g, (&z, &z, &z), (&z, &z), (1, 2, 2, 0, 4)).is_err());
    assert!(attention_gqa(&g, (&z, &z, &z), (&z, &z), (1, 2, 2, 2, 0)).is_err());
    assert!(
        attention_gqa(&g, (&z, &z, &z), (&z, &z), (1, 3, 2, 2, 4)).is_err(),
        "buffer sizes are checked"
    );
    let kc = g.zeros(4 * 8).unwrap();
    let (q, pr, o) = (
        g.zeros(16).unwrap(),
        g.zeros(16).unwrap(),
        g.zeros(16).unwrap(),
    );
    let go = |heads, kv, len| {
        attention_decode_gqa(&g, (&q, &kc, &kc), (&pr, &o), (heads, kv, 4, len, 4))
    };
    assert!(go(4, 2, 0).unwrap_err().to_string().contains("len 0"));
    assert!(go(4, 2, 5).is_err());
    assert!(go(4, 3, 2).is_err());
    assert!(go(4, 2, 4).is_ok());
}

#[test]
fn multi_command_ops_fail_cleanly_at_every_command() {
    let Some(g) = open_for_test() else { return };
    let (batch, t, heads, kv, hd) = (1, 3, 4, 2, 4);
    let (q, k) = (
        g.zeros(t * heads * hd).unwrap(),
        g.zeros(t * kv * hd).unwrap(),
    );
    let (p, o) = (
        g.zeros(heads * t * t).unwrap(),
        g.zeros(t * heads * hd).unwrap(),
    );
    assert_eq!(
        g.sweep_faults(|| attention_gqa(&g, (&q, &k, &k), (&p, &o), (batch, t, heads, kv, hd))),
        3
    );
    let (qd, kc) = (g.zeros(heads * hd).unwrap(), g.zeros(4 * kv * hd).unwrap());
    let (pr, od) = (g.zeros(heads * 4).unwrap(), g.zeros(heads * hd).unwrap());
    assert_eq!(
        g.sweep_faults(|| attention_decode_gqa(
            &g,
            (&qd, &kc, &kc),
            (&pr, &od),
            (heads, kv, hd, 2, 4)
        )),
        2
    );
    let (cs, sn) = (g.zeros(8).unwrap(), g.zeros(8).unwrap());
    let x = g.zeros(2 * heads * hd).unwrap();
    let xo = g.zeros(2 * heads * hd).unwrap();
    assert_eq!(
        g.sweep_faults(|| rope(&g, &x, (&cs, &sn), (2, heads, hd, 4), RopeType::Neox, 0)),
        1
    );
    assert_eq!(g.sweep_faults(|| swiglu(&g, &x, &x, &xo)), 1);
    let (gain, y) = (g.zeros(4).unwrap(), g.zeros(8).unwrap());
    let x8 = g.zeros(8).unwrap();
    assert_eq!(
        g.sweep_faults(|| rmsnorm(&g, &x8, &gain, &y, (2, 4), 1e-5)),
        1
    );
    let (ids, out) = (g.upload_indices(&[0, 1], 2).unwrap(), g.zeros(8).unwrap());
    assert_eq!(
        g.sweep_faults(|| gather_rows(&g, &x8, &ids, &out, (2, 4))),
        1
    );
}
