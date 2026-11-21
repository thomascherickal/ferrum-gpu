use super::*;

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
