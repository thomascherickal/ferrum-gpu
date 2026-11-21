use super::*;

#[test]
pub(super) fn rmsnorm_bwd_matches_finite_differences() {
    let (rows, cols) = (3, 5);
    let (x, gain, dy) = (
        rand_vec(rows * cols, 1),
        rand_vec(cols, 2),
        rand_vec(rows * cols, 3),
    );
    let (dx, dg) = rmsnorm_bwd(&x, &gain, &dy, (rows, cols), 1e-5);
    check_fd("dx", &x, &dx, &|x| {
        dot(&rmsnorm(x, &gain, rows, cols, 1e-5), &dy)
    });
    check_fd("dgain", &gain, &dg, &|g| {
        dot(&rmsnorm(&x, g, rows, cols, 1e-5), &dy)
    });
}

#[test]
pub(super) fn swiglu_bwd_matches_finite_differences_and_saturates() {
    let (gate, up, dact) = (rand_vec(12, 4), rand_vec(12, 5), rand_vec(12, 6));
    let (dg, du) = swiglu_bwd(&gate, &up, &dact);
    check_fd("dgate", &gate, &dg, &|g| dot(&swiglu(g, &up), &dact));
    check_fd("dup", &up, &du, &|u| dot(&swiglu(&gate, u), &dact));
    let (dg, du) = swiglu_bwd(&[-200.0, 200.0, 0.0], &[1.0; 3], &[1.0; 3]);
    assert!(
        dg.iter().chain(&du).all(|v| v.is_finite()),
        "saturation must not give NaN"
    );
    assert_eq!(dg[0], 0.0);
    assert!((dg[1] - 1.0).abs() < 1e-6 && (dg[2] - 0.5).abs() < 1e-6);
}

#[test]
pub(super) fn rope_bwd_is_the_transpose_of_rope_for_both_pairings() {
    for ty in [RopeType::Norm, RopeType::Neox] {
        let (rows, heads, hd, rd) = (3, 2, 6, 4);
        let sc = RopeScaling {
            freq_factors: vec![1.0, 3.0],
            linear_factor: 1.5,
        };
        let (c, s) = rope_tables(2, rows, rd, 10000.0, &sc);
        let (x, y) = (
            rand_vec(rows * heads * hd, 7),
            rand_vec(rows * heads * hd, 8),
        );
        let mut rx = x.clone();
        apply_rope(&mut rx, (rows, heads, hd, rd), (&c, &s), ty);
        let mut ty_ = y.clone();
        rope_bwd(&mut ty_, (rows, heads, hd, rd), (&c, &s), ty);
        // <R x, y> == <x, R^T y>
        assert!((dot(&rx, &y) - dot(&x, &ty_)).abs() < 1e-4, "{ty:?}");
        // and R^T R = I
        let mut back = rx.clone();
        rope_bwd(&mut back, (rows, heads, hd, rd), (&c, &s), ty);
        assert!(
            back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-5),
            "{ty:?}"
        );
    }
}
