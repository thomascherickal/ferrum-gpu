use super::*;
use crate::device::open_for_test;
use crate::ops_llama::rope;
use crate::tolerance::{max_rel_err, BACKWARD};
use ferrum_ref::llama as cpu;
use ferrum_ref::llama::{RopeScaling, RopeType};
use ferrum_ref::matmul::rand_vec;

fn show(name: &str, shape: &str, e: f32) {
    eprintln!("{name} {shape}: max rel err {e:.2e} (tol {BACKWARD:.0e})");
    assert!(
        e < BACKWARD,
        "{name} {shape}: max rel err {e} >= {BACKWARD}"
    );
}

fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
    rand_vec(n, seed).into_iter().map(|v| v * by).collect()
}

#[test]
fn rmsnorm_bwd_matches_cpu_on_odd_shapes() {
    let Some(g) = open_for_test() else { return };
    for (rows, cols) in [
        (1usize, 1usize),
        (1, 7),
        (5, 3),
        (64, 96),
        (65, 33),
        (3, 1000),
    ] {
        let (x, gain, dy) = (
            spread(rows * cols, 1, 3.0),
            rand_vec(cols, 2),
            rand_vec(rows * cols, 3),
        );
        let (gx, gg, gdy) = (
            g.upload(&x).unwrap(),
            g.upload(&gain).unwrap(),
            g.upload(&dy).unwrap(),
        );
        let (dx, dg, st) = (
            g.zeros(rows * cols).unwrap(),
            g.zeros(cols).unwrap(),
            g.zeros(rows).unwrap(),
        );
        rmsnorm_bwd(&g, (&gx, &gg, &gdy), (&dx, &dg), &st, (rows, cols), 1e-5).unwrap();
        let (wdx, wdg) = cpu::rmsnorm_bwd(&x, &gain, &dy, (rows, cols), 1e-5);
        show(
            "rmsnorm_bwd dx",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&dx).unwrap(), &wdx),
        );
        show(
            "rmsnorm_bwd dgain",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&dg).unwrap(), &wdg),
        );
    }
}

#[test]
fn swiglu_bwd_matches_cpu_including_saturation() {
    let Some(g) = open_for_test() else { return };
    let mut gate = spread(1000, 4, 6.0);
    gate.extend([0.0, 20.0, -20.0, 100.0, -100.0, 1e6, -1e6, 87.0, -89.0]);
    let (up, dact) = (spread(gate.len(), 5, 2.0), spread(gate.len(), 6, 2.0));
    let n = gate.len();
    let (gg, gu, gd) = (
        g.upload(&gate).unwrap(),
        g.upload(&up).unwrap(),
        g.upload(&dact).unwrap(),
    );
    let (dgate, dup) = (g.zeros(n).unwrap(), g.zeros(n).unwrap());
    swiglu_bwd(&g, (&gg, &gu, &gd), (&dgate, &dup)).unwrap();
    let (got_g, got_u) = (g.read(&dgate).unwrap(), g.read(&dup).unwrap());
    assert!(
        got_g.iter().chain(&got_u).all(|v| v.is_finite()),
        "no NaN/inf for huge inputs"
    );
    let (wg, wu) = cpu::swiglu_bwd(&gate, &up, &dact);
    show("swiglu_bwd dgate", "mixed", max_rel_err(&got_g, &wg));
    show("swiglu_bwd dup", "mixed", max_rel_err(&got_u, &wu));
}

#[test]
fn rope_with_a_negated_sine_table_is_the_cpu_backward() {
    let Some(g) = open_for_test() else { return };
    for ty in [RopeType::Norm, RopeType::Neox] {
        let (rows, heads, hd, rd, off) = (5usize, 3usize, 8usize, 6usize, 2usize);
        let sc = RopeScaling {
            freq_factors: vec![1.0, 2.0, 4.0],
            linear_factor: 1.5,
        };
        let (c, s) = cpu::rope_tables(0, off + rows, rd, 10000.0, &sc);
        let neg: Vec<f32> = s.iter().map(|v| -v).collect();
        let dy = spread(rows * heads * hd, 7, 2.0);
        let half = rd / 2;
        let mut want = dy.clone();
        cpu::rope_bwd(
            &mut want,
            (rows, heads, hd, rd),
            (&c[off * half..], &s[off * half..]),
            ty,
        );
        let (gdy, gc, gn) = (
            g.upload(&dy).unwrap(),
            g.upload(&c).unwrap(),
            g.upload(&neg).unwrap(),
        );
        rope(&g, &gdy, (&gc, &gn), (rows, heads, hd, rd), ty, off).unwrap();
        show(
            "rope backward",
            &format!("{ty:?}"),
            max_rel_err(&g.read(&gdy).unwrap(), &want),
        );
    }
}

#[test]
fn gqa_attention_bwd_matches_cpu_for_every_group_ratio() {
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
