use super::*;

#[test]
fn matmul_ex_matches_cpu_for_all_four_transpose_combinations() {
    let Some(g) = open_for_test() else { return };
    for (m, k, n) in [
        (1usize, 1usize, 1usize),
        (5, 7, 3),
        (17, 33, 15),
        (64, 64, 64),
        (65, 17, 130),
    ] {
        for (at, bt) in [(false, false), (true, false), (false, true), (true, true)] {
            let a = rand_vec(m * k, 1);
            let b = rand_vec(k * n, 2);
            let a_eff = if at {
                grad::transpose(&a, k, m)
            } else {
                a.clone()
            };
            let b_eff = if bt {
                grad::transpose(&b, n, k)
            } else {
                b.clone()
            };
            let want = matmul(&a_eff, &b_eff, m, k, n).unwrap();
            let (ga, gb, gc) = (
                g.upload(&a).unwrap(),
                g.upload(&b).unwrap(),
                g.zeros(m * n).unwrap(),
            );
            matmul_ex(&g, (&ga, &gb, &gc), (m, k, n), (at, bt)).unwrap();
            show(
                "matmul_ex",
                &format!("{m}x{k}x{n} at={at} bt={bt}"),
                max_rel_err(&g.read(&gc).unwrap(), &want),
                1e-4,
            );
        }
    }
}

#[test]
fn colsum_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for (rows, cols) in [(1usize, 1usize), (7, 3), (128, 33), (2048, 96)] {
        let x = rand_vec(rows * cols, 3);
        let mut want = vec![0.0f32; cols];
        for r in 0..rows {
            for c in 0..cols {
                want[c] += x[r * cols + c];
            }
        }
        let (gx, go) = (g.upload(&x).unwrap(), g.zeros(cols).unwrap());
        colsum(&g, &gx, &go, (rows, cols)).unwrap();
        show(
            "colsum",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&go).unwrap(), &want),
            REDUCING,
        );
    }
}

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
