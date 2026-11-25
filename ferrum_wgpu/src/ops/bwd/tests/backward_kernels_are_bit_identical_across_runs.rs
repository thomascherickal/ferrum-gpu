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
    let run = || gpu_layernorm_bwd(&g, (&x, &gain, &dy), (rows, cols));
    assert_eq!(run(), run());
}
