use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, EXACT, REDUCING};
use ferrum_ref::matmul::rand_vec;
use ferrum_ref::nn;

fn show(name: &str, shape: &str, e: f32, tol: f32) {
    eprintln!("{name} {shape}: max rel err {e:.2e} (tol {tol:.0e})");
    assert!(e < tol, "{name} {shape}: max rel err {e} >= {tol}");
}

/// Scale random data so GELU/softmax see a useful range, including large magnitudes.
fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
    rand_vec(n, seed).into_iter().map(|v| v * by).collect()
}

#[test]
fn add_and_bias_match_cpu() {
    let Some(g) = open_for_test() else { return };
    for n in [1usize, 7, 64, 65, 1000] {
        let (a, b) = (rand_vec(n, 1), rand_vec(n, 2));
        let (ga, gb, gc) = (
            g.upload(&a).unwrap(),
            g.upload(&b).unwrap(),
            g.zeros(n).unwrap(),
        );
        add(&g, &ga, &gb, &gc).unwrap();
        show(
            "add",
            &format!("n={n}"),
            max_rel_err(&g.read(&gc).unwrap(), &nn::add(&a, &b)),
            EXACT,
        );
    }
    for (rows, cols) in [(1usize, 1usize), (3, 5), (17, 64), (2, 100)] {
        let (x, bias) = (rand_vec(rows * cols, 3), rand_vec(cols, 4));
        let (gx, gb, gy) = (
            g.upload(&x).unwrap(),
            g.upload(&bias).unwrap(),
            g.zeros(rows * cols).unwrap(),
        );
        add_bias(&g, &gx, &gb, &gy).unwrap();
        show(
            "add_bias",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&gy).unwrap(), &nn::add_bias(&x, &bias)),
            EXACT,
        );
    }
}

#[test]
fn one_d_kernels_cover_more_than_a_single_dispatch_row() {
    let Some(g) = open_for_test() else { return };
    // 4,194,240 is the old cap (65535 groups x 64). Go past it so the y dimension is used.
    let n = 4_194_240 + 1_000;
    let a: Vec<f32> = (0..n).map(|i| (i % 1000) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5).collect();
    let (ga, gb, gc) = (
        g.upload(&a).unwrap(),
        g.upload(&b).unwrap(),
        g.zeros(n).unwrap(),
    );
    add(&g, &ga, &gb, &gc).unwrap();
    let got = g.read(&gc).unwrap();
    for i in [0, 1, 4_194_239, 4_194_240, 4_194_241, n - 1] {
        assert_eq!(got[i], a[i] + b[i], "element {i}");
    }
    assert_eq!(
        got,
        nn::add(&a, &b),
        "every element, including the second dispatch row"
    );
}

#[test]
fn gelu_matches_cpu_including_saturation() {
    let Some(g) = open_for_test() else { return };
    let mut x = spread(1000, 5, 6.0);
    x.extend([0.0, 20.0, -20.0, 100.0, -100.0, 1e6, -1e6, 10.0, -10.0]);
    let (gx, gy) = (g.upload(&x).unwrap(), g.zeros(x.len()).unwrap());
    gelu(&g, &gx, &gy).unwrap();
    let got = g.read(&gy).unwrap();
    assert!(
        got.iter().all(|v| v.is_finite()),
        "no NaN/inf for huge inputs"
    );
    let want: Vec<f32> = x.iter().map(|&v| nn::gelu(v)).collect();
    show("gelu", "mixed", max_rel_err(&got, &want), REDUCING);
    for (i, want) in [
        (0usize, 0.0f32),
        (1, 20.0),
        (2, 0.0),
        (3, 100.0),
        (4, 0.0),
        (5, 1e6),
        (6, 0.0),
    ] {
        let v = got[1000 + i];
        assert!(
            (v - want).abs() <= 1e-6 * want.abs().max(1.0),
            "saturated value {i}: {v}"
        );
    }
}

#[test]
fn layernorm_matches_cpu_on_odd_shapes() {
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
        let (gain, shift) = (rand_vec(cols, 7), rand_vec(cols, 8));
        let (gx, gg, gs, gy) = (
            g.upload(&x).unwrap(),
            g.upload(&gain).unwrap(),
            g.upload(&shift).unwrap(),
            g.zeros(rows * cols).unwrap(),
        );
        layernorm(&g, &gx, &gg, &gs, &gy, (rows, cols)).unwrap();
        let want = nn::layernorm(&x, &gain, &shift, rows, cols);
        show(
            "layernorm",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&gy).unwrap(), &want),
            REDUCING,
        );
    }
}

#[test]
fn attention_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for shape in [
        (1usize, 1usize, 1usize, 1usize),
        (1, 1, 2, 4),
        (2, 5, 2, 3),
        (1, 17, 4, 8),
        (2, 33, 3, 5),
    ] {
        let (batch, t, heads, hd) = shape;
        let n = batch * t * heads * hd;
        let (q, k, v) = (spread(n, 1, 2.0), spread(n, 2, 2.0), rand_vec(n, 3));
        let (gq, gk, gv) = (
            g.upload(&q).unwrap(),
            g.upload(&k).unwrap(),
            g.upload(&v).unwrap(),
        );
        let (gp, go) = (g.zeros(batch * heads * t * t).unwrap(), g.zeros(n).unwrap());
        attention(&g, (&gq, &gk, &gv), (&gp, &go), shape).unwrap();
        let (want_out, want_probs) = nn::attention(&q, &k, &v, shape);
        show(
            "attention probs",
            &format!("{shape:?}"),
            max_rel_err(&g.read(&gp).unwrap(), &want_probs),
            REDUCING,
        );
        show(
            "attention out",
            &format!("{shape:?}"),
            max_rel_err(&g.read(&go).unwrap(), &want_out),
            REDUCING,
        );
    }
}

#[test]
fn cross_entropy_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for (rows, vocab) in [(1usize, 2usize), (3, 5), (16, 1024), (33, 7)] {
        let logits = spread(rows * vocab, 9, 8.0);
        let targets: Vec<usize> = (0..rows).map(|i| (i * 7 + 1) % vocab).collect();
        let (gl, gt) = (
            g.upload(&logits).unwrap(),
            g.upload_indices(&targets, vocab).unwrap(),
        );
        let (gg, gr, gloss) = (
            g.zeros(rows * vocab).unwrap(),
            g.zeros(rows).unwrap(),
            g.zeros(1).unwrap(),
        );
        cross_entropy(&g, &gl, &gt, (&gg, &gr, &gloss), (rows, vocab)).unwrap();
        let (want_loss, want_grad) = nn::cross_entropy(&logits, &targets, vocab);
        show(
            "ce loss",
            &format!("{rows}x{vocab}"),
            max_rel_err(&g.read(&gloss).unwrap(), &[want_loss]),
            REDUCING,
        );
        show(
            "ce grad",
            &format!("{rows}x{vocab}"),
            max_rel_err(&g.read(&gg).unwrap(), &want_grad),
            REDUCING,
        );
    }
}

#[test]
fn score_matches_cpu_including_ties() {
    let Some(g) = open_for_test() else { return };
    let (rows, vocab) = (5usize, 6usize);
    let mut logits = spread(rows * vocab, 11, 4.0);
    logits[vocab..2 * vocab].fill(1.5); // a fully tied row: first index is the argmax
    let targets = vec![0usize, 0, 3, 5, 1];
    let (gl, gt, go) = (
        g.upload(&logits).unwrap(),
        g.upload_indices(&targets, vocab).unwrap(),
        g.zeros(2 * rows).unwrap(),
    );
    score(&g, &gl, &gt, &go, (rows, vocab)).unwrap();
    let got = g.read(&go).unwrap();
    let want = nn::score(&logits, &targets, vocab);
    for (i, (lp, hit)) in want.iter().enumerate() {
        let g0 = got[2 * i];
        assert!(
            (g0 - lp).abs() < 1e-4 * lp.abs().max(1.0),
            "row {i}: {g0} vs {lp}"
        );
        assert_eq!(got[2 * i + 1] == 1.0, *hit, "row {i} argmax flag");
    }
    assert_eq!(got[3], 1.0, "tied row with target 0 counts as argmax");
}

#[test]
fn embed_matches_cpu_and_rejects_bad_ids() {
    let Some(g) = open_for_test() else { return };
    let (vocab, c, t) = (11usize, 6usize, 4usize);
    let (tok, pos) = (rand_vec(vocab * c, 12), rand_vec(t * c, 13));
    let ids = vec![0usize, 10, 3, 3, 7, 1, 2]; // 7 rows, two sequences' worth of positions
    let (gt, gp, gi) = (
        g.upload(&tok).unwrap(),
        g.upload(&pos).unwrap(),
        g.upload_indices(&ids, vocab).unwrap(),
    );
    let go = g.zeros(ids.len() * c).unwrap();
    embed(&g, (&gt, &gp, &gi), &go, (ids.len(), c, t)).unwrap();
    show(
        "embed",
        "7x6",
        max_rel_err(&g.read(&go).unwrap(), &nn::embed(&tok, &pos, &ids, t, c)),
        EXACT,
    );
    let e = g
        .upload_indices(&[3, 11], vocab)
        .expect_err("id 11 is out of range");
    assert!(e.to_string().contains("11"));
}

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

#[test]
fn shape_mismatches_are_errors_not_dispatches() {
    let Some(g) = open_for_test() else { return };
    let (a, b, c) = (
        g.zeros(4).unwrap(),
        g.zeros(5).unwrap(),
        g.zeros(4).unwrap(),
    );
    assert!(add(&g, &a, &b, &c)
        .unwrap_err()
        .to_string()
        .contains("expected 4"));
    assert!(add_bias(&g, &a, &g.zeros(3).unwrap(), &c).is_err());
    assert!(add_bias(&g, &a, &g.zeros(0).unwrap(), &c).is_err());
    assert!(layernorm(
        &g,
        &a,
        &g.zeros(2).unwrap(),
        &g.zeros(2).unwrap(),
        &c,
        (2, 3)
    )
    .is_err());
    let (l, t) = (g.zeros(6).unwrap(), g.upload_indices(&[0, 1], 3).unwrap());
    let one = g.zeros(1).unwrap();
    assert!(cross_entropy(
        &g,
        &l,
        &t,
        (&g.zeros(6).unwrap(), &g.zeros(2).unwrap(), &one),
        (0, 3)
    )
    .is_err());
    assert!(embed(
        &g,
        (&g.zeros(4).unwrap(), &g.zeros(2).unwrap(), &t),
        &g.zeros(8).unwrap(),
        (2, 4, 4)
    )
    .is_err());
}

#[test]
fn gemv_matches_cpu_linear() {
    let Some(g) = open_for_test() else { return };
    for (k, n) in [(1usize, 1usize), (7, 5), (96, 64), (65, 130), (320, 96)] {
        let (x, w, b) = (rand_vec(k, 1), rand_vec(k * n, 2), rand_vec(n, 3));
        let want = nn::add_bias(&ferrum_ref::matmul::matmul(&x, &w, 1, k, n).unwrap(), &b);
        let (gx, gw, gb, gy) = (
            g.upload(&x).unwrap(),
            g.upload(&w).unwrap(),
            g.upload(&b).unwrap(),
            g.zeros(n).unwrap(),
        );
        gemv(&g, (&gx, &gw, &gb), &gy, (k, n)).unwrap();
        show(
            "gemv",
            &format!("{k}x{n}"),
            max_rel_err(&g.read(&gy).unwrap(), &want),
            REDUCING,
        );
    }
    let z = g.zeros(2).unwrap();
    assert!(gemv(&g, (&z, &g.zeros(3).unwrap(), &z,), &z, (2, 2)).is_err());
}

#[test]
fn attention_decode_matches_the_last_row_of_full_attention() {
    let Some(g) = open_for_test() else { return };
    let (heads, hd, t, ctx) = (3usize, 5usize, 9usize, 12usize);
    let c = heads * hd;
    let (q, k, v) = (
        spread(t * c, 31, 2.0),
        spread(t * c, 32, 2.0),
        rand_vec(t * c, 33),
    );
    let (want_out, _) = nn::attention(&q, &k, &v, (1, t, heads, hd));
    for len in [1usize, 4, t] {
        // Cache holds positions 0..len; the query is position len-1.
        let mut kc = vec![0.0; ctx * c];
        let mut vc = vec![0.0; ctx * c];
        kc[..len * c].copy_from_slice(&k[..len * c]);
        vc[..len * c].copy_from_slice(&v[..len * c]);
        let (gq, gk, gv) = (
            g.upload(&q[(len - 1) * c..len * c]).unwrap(),
            g.upload(&kc).unwrap(),
            g.upload(&vc).unwrap(),
        );
        let (gp, go) = (g.zeros(heads * ctx).unwrap(), g.zeros(c).unwrap());
        attention_decode(&g, (&gq, &gk, &gv), (&gp, &go), (heads, hd, len, ctx)).unwrap();
        let want = &want_out[(len - 1) * c..len * c];
        show(
            "attention_decode",
            &format!("len={len}"),
            max_rel_err(&g.read(&go).unwrap(), want),
            REDUCING,
        );
    }
    let z = g.zeros(c).unwrap();
    let big = g.zeros(ctx * c).unwrap();
    let pr = g.zeros(heads * ctx).unwrap();
    assert!(attention_decode(&g, (&z, &big, &big), (&pr, &z), (heads, hd, 0, ctx)).is_err());
    assert!(attention_decode(&g, (&z, &big, &big), (&pr, &z), (heads, hd, ctx + 1, ctx)).is_err());
}

#[test]
fn embed_at_uses_the_absolute_position() {
    let Some(g) = open_for_test() else { return };
    let (vocab, c, t) = (6usize, 4usize, 5usize);
    let (tok, pos) = (rand_vec(vocab * c, 41), rand_vec(t * c, 42));
    for p in [0usize, 3, 4] {
        let gi = g.upload_indices(&[2], vocab).unwrap();
        let (gt, gp, go) = (
            g.upload(&tok).unwrap(),
            g.upload(&pos).unwrap(),
            g.zeros(c).unwrap(),
        );
        embed_at(&g, (&gt, &gp, &gi), &go, (1, c, t), p).unwrap();
        let want: Vec<f32> = (0..c).map(|d| tok[2 * c + d] + pos[p * c + d]).collect();
        show(
            "embed_at",
            &format!("pos={p}"),
            max_rel_err(&g.read(&go).unwrap(), &want),
            EXACT,
        );
    }
}

#[test]
fn repeat_runs_are_bit_identical() {
    let Some(g) = open_for_test() else { return };
    let (rows, vocab) = (20usize, 50usize);
    let logits = spread(rows * vocab, 21, 5.0);
    let targets: Vec<usize> = (0..rows).map(|i| i % vocab).collect();
    let run = || {
        let (gl, gt) = (
            g.upload(&logits).unwrap(),
            g.upload_indices(&targets, vocab).unwrap(),
        );
        let (gg, gr, gloss) = (
            g.zeros(rows * vocab).unwrap(),
            g.zeros(rows).unwrap(),
            g.zeros(1).unwrap(),
        );
        cross_entropy(&g, &gl, &gt, (&gg, &gr, &gloss), (rows, vocab)).unwrap();
        (g.read(&gg).unwrap(), g.read(&gloss).unwrap())
    };
    assert_eq!(run(), run());
}
