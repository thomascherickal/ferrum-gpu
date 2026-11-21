use super::*;

#[test]
fn rmsnorm_bwd_matches_cpu_on_odd_shapes() {
    let Some(g) = open_for_test() else { return };
    for (rows, cols) in [
        (1usize, 1usize),
        (1, 7),
        (5, 3),
        (64, 96),
        (65, 33),
        (3, 1000),
    ] {
        let (x, gain, dy) = (
            spread(rows * cols, 1, 3.0),
            rand_vec(cols, 2),
            rand_vec(rows * cols, 3),
        );
        let (gx, gg, gdy) = (
            g.upload(&x).unwrap(),
            g.upload(&gain).unwrap(),
            g.upload(&dy).unwrap(),
        );
        let (dx, dg, st) = (
            g.zeros(rows * cols).unwrap(),
            g.zeros(cols).unwrap(),
            g.zeros(rows).unwrap(),
        );
        rmsnorm_bwd(&g, (&gx, &gg, &gdy), (&dx, &dg), &st, (rows, cols), 1e-5).unwrap();
        let (wdx, wdg) = cpu::rmsnorm_bwd(&x, &gain, &dy, (rows, cols), 1e-5);
        show(
            "rmsnorm_bwd dx",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&dx).unwrap(), &wdx),
        );
        show(
            "rmsnorm_bwd dgain",
            &format!("{rows}x{cols}"),
            max_rel_err(&g.read(&dg).unwrap(), &wdg),
        );
    }
}

#[test]
fn swiglu_bwd_matches_cpu_including_saturation() {
    let Some(g) = open_for_test() else { return };
    let mut gate = spread(1000, 4, 6.0);
    gate.extend([0.0, 20.0, -20.0, 100.0, -100.0, 1e6, -1e6, 87.0, -89.0]);
    let (up, dact) = (spread(gate.len(), 5, 2.0), spread(gate.len(), 6, 2.0));
    let n = gate.len();
    let (gg, gu, gd) = (
        g.upload(&gate).unwrap(),
        g.upload(&up).unwrap(),
        g.upload(&dact).unwrap(),
    );
    let (dgate, dup) = (g.zeros(n).unwrap(), g.zeros(n).unwrap());
    swiglu_bwd(&g, (&gg, &gu, &gd), (&dgate, &dup)).unwrap();
    let (got_g, got_u) = (g.read(&dgate).unwrap(), g.read(&dup).unwrap());
    assert!(
        got_g.iter().chain(&got_u).all(|v| v.is_finite()),
        "no NaN/inf for huge inputs"
    );
    let (wg, wu) = cpu::swiglu_bwd(&gate, &up, &dact);
    show("swiglu_bwd dgate", "mixed", max_rel_err(&got_g, &wg));
    show("swiglu_bwd dup", "mixed", max_rel_err(&got_u, &wu));
}
