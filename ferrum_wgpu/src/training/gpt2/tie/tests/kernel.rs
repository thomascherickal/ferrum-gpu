use crate::device::open_for_test;
use crate::ops_opt::transpose_into;
use ferrum_ref::grad::transpose;

/// A matrix of distinct values.
fn matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| i as f32 * 0.25 - 7.0).collect()
}

#[test]
fn transposes_exactly_for_shapes_off_the_workgroup_size() {
    let Some(g) = open_for_test() else { return };
    // (rows, cols) of the destination: 1-wide, prime, and larger than one workgroup of 64
    for (rows, cols) in [
        (1usize, 1usize),
        (1, 70),
        (70, 1),
        (7, 13),
        (24, 16),
        (16, 24),
        (37, 61),
    ] {
        let src = matrix(cols, rows); // [cols, rows]
        let dst = g.zeros(rows * cols).unwrap();
        transpose_into(&g, &g.upload(&src).unwrap(), &dst, (rows, cols), false).unwrap();
        assert_eq!(
            g.read(&dst).unwrap(),
            transpose(&src, cols, rows),
            "{rows}x{cols}"
        );
    }
}

#[test]
fn add_accumulates_into_what_is_there_and_overwrite_ignores_it() {
    let Some(g) = open_for_test() else { return };
    let (rows, cols) = (5, 9);
    let src = matrix(cols, rows);
    let base = matrix(rows, cols);
    let want_t = transpose(&src, cols, rows);
    let dst = g.upload(&base).unwrap();
    transpose_into(&g, &g.upload(&src).unwrap(), &dst, (rows, cols), true).unwrap();
    let want: Vec<f32> = base.iter().zip(&want_t).map(|(b, t)| b + t).collect();
    assert_eq!(g.read(&dst).unwrap(), want, "added");
    transpose_into(&g, &g.upload(&src).unwrap(), &dst, (rows, cols), false).unwrap();
    assert_eq!(g.read(&dst).unwrap(), want_t, "overwritten");
}

#[test]
fn a_buffer_of_the_wrong_length_is_refused_by_name() {
    let Some(g) = open_for_test() else { return };
    let (ok, short) = (g.zeros(12).unwrap(), g.zeros(11).unwrap());
    assert!(transpose_into(&g, &ok, &short, (3, 4), false)
        .unwrap_err()
        .to_string()
        .contains("transpose dst"));
    assert!(transpose_into(&g, &short, &ok, (3, 4), false)
        .unwrap_err()
        .to_string()
        .contains("transpose src"));
}
