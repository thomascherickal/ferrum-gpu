use super::*;

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
