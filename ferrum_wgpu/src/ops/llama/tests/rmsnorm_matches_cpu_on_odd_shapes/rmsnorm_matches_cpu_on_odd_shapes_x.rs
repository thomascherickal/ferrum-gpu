use super::*;

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
