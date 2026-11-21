use super::*;

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
