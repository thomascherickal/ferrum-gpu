use super::*;
use crate::matmul::{matmul, rand_vec};

#[test]
fn bit_identical_to_the_simple_kernel_on_awkward_shapes_and_thread_counts() {
    // Shapes straddle the 4x8 register tile and the 256 cache panels.
    for &(m, k, n) in &[
        (1, 1, 1),
        (3, 5, 7),
        (4, 8, 8),
        (5, 9, 17),
        (9, 300, 11),
        (13, 513, 261),
        (64, 64, 64),
    ] {
        let (a, b) = (rand_vec(m * k, 1), rand_vec(k * n, 2));
        let want = matmul(&a, &b, m, k, n).unwrap();
        for t in [1, 2, 3, 8, 100] {
            assert_eq!(
                matmul_tuned(&a, &b, m, k, n, t).unwrap(),
                want,
                "{m}x{k}x{n} threads {t}"
            );
        }
    }
}

#[test]
fn empty_and_mismatched_inputs() {
    assert!(matmul_tuned(&[], &[], 0, 3, 0, 4).unwrap().is_empty());
    assert_eq!(matmul_tuned(&[], &[], 2, 0, 2, 2).unwrap(), vec![0.0; 4]);
    assert!(matmul_tuned(&[1.0], &[1.0, 2.0], 1, 2, 1, 1).is_err());
    assert!(matmul_tuned(&[1.0, 2.0], &[1.0], 1, 2, 1, 1).is_err());
}
