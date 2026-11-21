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
