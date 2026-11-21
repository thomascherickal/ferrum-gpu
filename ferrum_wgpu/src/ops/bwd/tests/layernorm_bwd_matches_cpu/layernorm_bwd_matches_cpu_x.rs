use super::*;

#[test]
fn layernorm_bwd_matches_cpu() {
    let Some(g) = open_for_test() else { return };
    for (rows, cols) in [(1usize, 2usize), (5, 3), (64, 96), (33, 33), (3, 500)] {
        let x = spread(rows * cols, 9, 3.0);
        let (gain, dy) = (rand_vec(cols, 10), rand_vec(rows * cols, 11));
        let (wdx, wdg, wdb) = grad::layernorm_bwd(&x, &gain, &dy, rows, cols);
        let (gx, gg, gdy) = (
            g.upload(&x).unwrap(),
            g.upload(&gain).unwrap(),
            g.upload(&dy).unwrap(),
        );
        let (gdx, gdg, gdb, st) = (
            g.zeros(rows * cols).unwrap(),
            g.zeros(cols).unwrap(),
            g.zeros(cols).unwrap(),
            g.zeros(2 * rows).unwrap(),
        );
        layernorm_bwd(&g, (&gx, &gg, &gdy), (&gdx, &gdg, &gdb), &st, (rows, cols)).unwrap();
        let s = format!("{rows}x{cols}");
        show(
            "layernorm_bwd dx",
            &s,
            max_rel_err(&g.read(&gdx).unwrap(), &wdx),
            REDUCING,
        );
        show(
            "layernorm_bwd dgain",
            &s,
            max_rel_err(&g.read(&gdg).unwrap(), &wdg),
            REDUCING,
        );
        show(
            "layernorm_bwd dshift",
            &s,
            max_rel_err(&g.read(&gdb).unwrap(), &wdb),
            REDUCING,
        );
    }
}
