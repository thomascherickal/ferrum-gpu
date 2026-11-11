use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, EXACT, REDUCING};
use ferrum_ref::llama as cpu;
use ferrum_ref::matmul::rand_vec;

fn show(name: &str, shape: &str, e: f32, tol: f32) {
    eprintln!("{name} {shape}: max rel err {e:.2e} (tol {tol:.0e})");
    assert!(e < tol, "{name} {shape}: max rel err {e} >= {tol}");
}

fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
    rand_vec(n, seed).into_iter().map(|v| v * by).collect()
}

#[test]
fn rmsnorm_matches_cpu_on_odd_shapes() {
    let Some(g) = open_for_test() else { return };
    for (rows, cols) in [
        (1usize, 1usize),
        (1, 7),
        (5, 3),
        (64, 96),
        (65, 33),
        (3, 1000),
    ] {
        let x = spread(rows * cols, 6, 3.0);
        let gain = rand_vec(cols, 7);
        let (gx, gg, gy) = (
            g.upload(&x).unwrap(),
            g.upload(&gain).unwrap(),
            g.zeros(rows * cols).unwrap(),
        );
        rmsnorm(&g, &gx, &gg, &gy, (rows, cols), 1e-5).unwrap();
        let want = cpu::rmsnorm(&x, &gain, rows, cols, 1e-5);
        show(
            "rmsnorm",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&gy).unwrap(), &want),
            REDUCING,
        );
    }
    // eps matters: an all-zero row stays zero and a large eps shrinks the output
    let z = g.zeros(4).unwrap();
    let (one, y) = (g.upload(&[1.0; 4]).unwrap(), g.zeros(4).unwrap());
    rmsnorm(&g, &z, &one, &y, (1, 4), 1e-5).unwrap();
    assert_eq!(g.read(&y).unwrap(), vec![0.0; 4]);
    let x = g.upload(&[1.0; 4]).unwrap();
    rmsnorm(&g, &x, &one, &y, (1, 4), 3.0).unwrap();
    assert!((g.read(&y).unwrap()[0] - 0.5).abs() < 1e-6, "1/sqrt(1+3)");
}

#[test]
fn rope_matches_cpu_for_both_pairings_offsets_and_partial_rotary() {
    let Some(g) = open_for_test() else { return };
    // (rows, heads, hd, rope_dim, offset)
    for (rows, heads, hd, rd, off) in [
        (1usize, 1usize, 2usize, 2usize, 0usize),
        (1, 2, 8, 8, 5),
        (4, 3, 8, 4, 0),
        (6, 4, 16, 16, 3),
        (3, 2, 10, 6, 7),
    ] {
        for ty in [RopeType::Norm, RopeType::Neox] {
            let scaling = cpu::RopeScaling {
                freq_factors: (0..rd / 2).map(|i| 1.0 + i as f32 * 0.5).collect(),
                linear_factor: 2.0,
            };
            let total = off + rows;
            let (c_all, s_all) = cpu::rope_tables(0, total, rd, 10000.0, &scaling);
            let x = spread(rows * heads * hd, 3, 2.0);
            let mut want = x.clone();
            let half = rd / 2;
            cpu::apply_rope(
                &mut want,
                (rows, heads, hd, rd),
                (&c_all[off * half..], &s_all[off * half..]),
                ty,
            );
            let (gx, gc, gs) = (
                g.upload(&x).unwrap(),
                g.upload(&c_all).unwrap(),
                g.upload(&s_all).unwrap(),
            );
            rope(&g, &gx, (&gc, &gs), (rows, heads, hd, rd), ty, off).unwrap();
            show(
                "rope",
                &format!("{ty:?} {rows}x{heads}x{hd} rd={rd} off={off}"),
                max_rel_err(&g.read(&gx).unwrap(), &want),
                EXACT,
            );
        }
    }
}

#[test]
fn rope_refuses_bad_shapes() {
    let Some(g) = open_for_test() else { return };
    let (x, t) = (g.zeros(8).unwrap(), g.zeros(4).unwrap());
    let go = |rd, rows, off| rope(&g, &x, (&t, &t), (rows, 1, 8, rd), RopeType::Norm, off);
    assert!(go(0, 1, 0).unwrap_err().to_string().contains("rope_dim"));
    assert!(go(3, 1, 0).is_err());
    assert!(go(10, 1, 0).is_err());
    assert!(
        go(4, 1, 0).is_ok(),
        "2 rows of the 4-element table cover row 0"
    );
    assert!(go(4, 1, 2)
        .unwrap_err()
        .to_string()
        .contains("do not cover"));
    assert!(go(4, 2, 1).is_err());
    let short = g.zeros(3).unwrap();
    assert!(rope(&g, &x, (&t, &short), (1, 1, 8, 4), RopeType::Norm, 0).is_err());
}

#[test]
fn swiglu_matches_cpu_including_saturation() {
    let Some(g) = open_for_test() else { return };
    let mut gate = spread(1000, 5, 6.0);
    gate.extend([
        0.0, 20.0, -20.0, 100.0, -100.0, 1e6, -1e6, 87.0, -87.0, -89.0,
    ]);
    let up = spread(gate.len(), 6, 2.0);
    let (gg, gu, gy) = (
        g.upload(&gate).unwrap(),
        g.upload(&up).unwrap(),
        g.zeros(gate.len()).unwrap(),
    );
    swiglu(&g, &gg, &gu, &gy).unwrap();
    let got = g.read(&gy).unwrap();
    assert!(
        got.iter().all(|v| v.is_finite()),
        "no NaN/inf for huge inputs"
    );
    show(
        "swiglu",
        "mixed",
        max_rel_err(&got, &cpu::swiglu(&gate, &up)),
        REDUCING,
    );
    assert!(swiglu(&g, &gg, &gu, &g.zeros(3).unwrap()).is_err());
}

#[test]
fn gather_rows_picks_rows_and_refuses_ragged_tables() {
    let Some(g) = open_for_test() else { return };
    let table: Vec<f32> = (0..15).map(|i| i as f32).collect(); // 5 rows of 3
    let (gt, ids, out) = (
        g.upload(&table).unwrap(),
        g.upload_indices(&[4, 0, 4, 2], 5).unwrap(),
        g.zeros(12).unwrap(),
    );
    gather_rows(&g, &gt, &ids, &out, (4, 3)).unwrap();
    assert_eq!(
        g.read(&out).unwrap(),
        vec![12.0, 13.0, 14.0, 0.0, 1.0, 2.0, 12.0, 13.0, 14.0, 6.0, 7.0, 8.0]
    );
    let out16 = g.zeros(16).unwrap();
    let e = gather_rows(&g, &gt, &ids, &out16, (4, 4)).unwrap_err();
    assert!(e.to_string().contains("whole rows"), "{e}");
    let zero_width = gather_rows(&g, &gt, &ids, &g.zeros(0).unwrap(), (4, 0));
    assert!(zero_width.unwrap_err().to_string().contains("whole rows"));
    assert!(gather_rows(&g, &gt, &ids, &g.zeros(3).unwrap(), (4, 3)).is_err());
    assert!(
        g.upload_indices(&[5], 5).is_err(),
        "out-of-range token refused on upload"
    );
}

/// CPU reference for full GQA attention: per batch, per row, `attend_row` over that row's prefix.
fn cpu_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (batch, t, heads, kv, hd): (usize, usize, usize, usize, usize),
) -> Vec<f32> {
    let (qd, kvd) = (heads * hd, kv * hd);
    let mut out = Vec::new();
    for b in 0..batch {
        for i in 0..t {
            let r = b * t + i;
            out.extend(cpu::attend_row(
                &q[r * qd..(r + 1) * qd],
                &k[b * t * kvd..],
                &v[b * t * kvd..],
                i + 1,
                (heads, kv, hd),
            ));
        }
    }
    out
}

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
