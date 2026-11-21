use super::*;

#[test]
fn known_values() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
    assert_eq!(
        matmul(&a, &b, 2, 3, 2).unwrap(),
        vec![58.0, 64.0, 139.0, 154.0]
    );
}

#[test]
fn non_square_and_one_row() {
    let a = rand_vec(7, 1);
    let b = rand_vec(7 * 5, 2);
    let got = matmul(&a, &b, 1, 7, 5).unwrap();
    for j in 0..5 {
        let want: f32 = (0..7).map(|p| a[p] * b[p * 5 + j]).sum();
        assert!((got[j] - want).abs() < 1e-5);
    }
}

#[test]
fn shape_errors_name_the_operand() {
    assert_eq!(
        matmul(&[1.0], &[1.0, 2.0], 2, 1, 1),
        Err(MatmulError::Len {
            which: "a",
            want: 2,
            got: 1
        })
    );
    let e = matmul(&[1.0], &[1.0, 2.0], 1, 1, 1).unwrap_err();
    assert!(e.to_string().contains("operand b"));
}

#[test]
fn empty_dimensions_give_empty_or_zero_output() {
    assert_eq!(matmul(&[], &[], 0, 3, 0).unwrap(), Vec::<f32>::new());
    assert_eq!(matmul(&[], &[], 2, 0, 2).unwrap(), vec![0.0; 4]);
    // The threaded variant takes the same early exit for empty outputs.
    assert!(matmul_threaded(&[], &[], 0, 3, 0, 4).unwrap().is_empty());
    assert!(matmul_threaded(&[], &[], 5, 0, 0, 2).unwrap().is_empty());
}

#[test]
fn threaded_is_bit_identical_for_any_thread_count() {
    let (m, k, n) = (13, 9, 11);
    let a = rand_vec(m * k, 3);
    let b = rand_vec(k * n, 4);
    let want = matmul(&a, &b, m, k, n).unwrap();
    for t in [0, 1, 2, 5, 64] {
        assert_eq!(
            matmul_threaded(&a, &b, m, k, n, t).unwrap(),
            want,
            "threads {t}"
        );
    }
    assert!(matmul_threaded(&a, &b, m, k + 1, n, 2).is_err());
}

#[test]
fn rand_vec_is_deterministic_and_in_range() {
    assert_eq!(rand_vec(8, 7), rand_vec(8, 7));
    assert_ne!(rand_vec(8, 7), rand_vec(8, 8));
    assert!(rand_vec(1000, 9).iter().all(|v| (-1.0..1.0).contains(v)));
}
