use super::*;

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
