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
