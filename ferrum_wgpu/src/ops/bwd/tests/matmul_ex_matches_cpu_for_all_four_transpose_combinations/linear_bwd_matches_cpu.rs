use super::*;

#[test]
fn linear_bwd_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for (m, k, n) in [
        (1usize, 1usize, 1usize),
        (6, 5, 7),
        (33, 20, 17),
        (128, 96, 64),
    ] {
        let (x, w, dy) = (rand_vec(m * k, 4), rand_vec(k * n, 5), rand_vec(m * n, 6));
        let (wdx, wdw, wdb) = grad::linear_bwd(&x, &w, &dy, (m, k, n), 1);
        let (gx, gw, gdy) = (
            g.upload(&x).unwrap(),
            g.upload(&w).unwrap(),
            g.upload(&dy).unwrap(),
        );
        let (gdx, gdw, gdb) = (
            g.zeros(m * k).unwrap(),
            g.zeros(k * n).unwrap(),
            g.zeros(n).unwrap(),
        );
        linear_bwd(&g, (&gx, &gw, &gdy), (&gdx, &gdw, &gdb), (m, k, n)).unwrap();
        let s = format!("{m}x{k}x{n}");
        show(
            "linear_bwd dx",
            &s,
            max_rel_err(&g.read(&gdx).unwrap(), &wdx),
            1e-4,
        );
        show(
            "linear_bwd dw",
            &s,
            max_rel_err(&g.read(&gdw).unwrap(), &wdw),
            1e-4,
        );
        show(
            "linear_bwd db",
            &s,
            max_rel_err(&g.read(&gdb).unwrap(), &wdb),
            REDUCING,
        );
    }
}

#[test]
fn gelu_bwd_matches_cpu_including_saturation() {
    let Some(g) = open_for_test() else { return };
    let mut f1 = spread(1000, 7, 8.0);
    f1.extend([0.0, 9.999, -9.999, 10.0, -10.0, 100.0, -100.0, 1e6, -1e6]);
    let dact = spread(f1.len(), 8, 1.0);
    let want: Vec<f32> = f1
        .iter()
        .zip(&dact)
        .map(|(&f, &d)| d * grad::gelu_grad(f))
        .collect();
    let (gd, gf, go) = (
        g.upload(&dact).unwrap(),
        g.upload(&f1).unwrap(),
        g.zeros(f1.len()).unwrap(),
    );
    gelu_bwd(&g, &gd, &gf, &go).unwrap();
    let got = g.read(&go).unwrap();
    assert!(got.iter().all(|v| v.is_finite()));
    show("gelu_bwd", "mixed", max_rel_err(&got, &want), REDUCING);
}
