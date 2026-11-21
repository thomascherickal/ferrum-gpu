use super::*;

#[test]
fn backward_kernels_are_bit_identical_across_runs() {
    let Some(g) = open_for_test() else { return };
    let (rows, cols) = (40usize, 24usize);
    let (x, gain, dy) = (
        spread(rows * cols, 17, 2.0),
        rand_vec(cols, 18),
        rand_vec(rows * cols, 19),
    );
    let run = || {
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
        (
            g.read(&gdx).unwrap(),
            g.read(&gdg).unwrap(),
            g.read(&gdb).unwrap(),
        )
    };
    assert_eq!(run(), run());
}
