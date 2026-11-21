use super::*;

#[test]
pub(super) fn linear_matches_the_cpu_matmul_on_the_dequantised_weights() {
    let Some(g) = open_for_test() else { return };
    for kind in EXT {
        for (k, n) in shapes(kind) {
            for rows in [1usize, 3, 17] {
                let x = rand_vec(rows * k, 5);
                let bytes = quantise_matrix(kind, &rand_vec(k * n, 21), k, n).unwrap();
                let w_rows = rows_of(kind, &bytes, k, n);
                let want = matmul(&x, &transpose(&w_rows, n, k), rows, k, n).unwrap();
                let qw = upload_qweight(&g, kind, &bytes, k, n).unwrap();
                let (gx, gy) = (g.upload(&x).unwrap(), g.zeros(rows * n).unwrap());
                linear_q(&g, &gx, &qw, &gy, rows).unwrap();
                let err = max_rel_err(&g.read(&gy).unwrap(), &want);
                assert!(err <= REDUCING, "{kind:?} {rows}x{k}x{n}: {err}");
            }
        }
    }
}

#[test]
pub(super) fn the_packed_weight_is_smaller_by_the_claimed_ratio() {
    let Some(g) = open_for_test() else { return };
    for (kind, k, ratio) in [
        (QKind::Fp4, 64, 128.0 / 20.0),
        (QKind::Fp8E4M3, 64, 128.0 / 36.0),
        (QKind::Fp8E5M2, 64, 128.0 / 36.0),
        (QKind::Q2K, 256, 1024.0 / 88.0),
    ] {
        let n = 8;
        let qw = upload_quantised(&g, kind, &rand_vec(k * n, 1), k, n).unwrap();
        let f32_bytes = (k * n * 4) as f64;
        assert!(
            (f32_bytes / qw.device_bytes() as f64 - ratio).abs() < 1e-9,
            "{kind:?}"
        );
        assert_eq!((qw.k(), qw.n(), qw.kind()), (k, n, kind));
    }
}

#[test]
pub(super) fn every_kernel_matches_the_cpu_matmul_for_the_extra_formats() {
    let Some(g) = open_for_test() else { return };
    for kind in EXT {
        for (k, n) in shapes(kind) {
            let bytes = quantise_matrix(kind, &rand_vec(k * n, 17), k, n).unwrap();
            let w_rows = rows_of(kind, &bytes, k, n);
            let qw = upload_qweight(&g, kind, &bytes, k, n).unwrap();
            for rows in [1usize, 3, 17, 33] {
                let x = rand_vec(rows * k, 5);
                let want = matmul(&x, &transpose(&w_rows, n, k), rows, k, n).unwrap();
                let gx = g.upload(&x).unwrap();
                let paths: &[QPath] = if rows == 1 {
                    &[QPath::Naive, QPath::Gemv, QPath::Tiled, QPath::Auto]
                } else {
                    &[QPath::Naive, QPath::Tiled, QPath::Auto]
                };
                for &path in paths {
                    let gy = g.zeros(rows * n).unwrap();
                    linear_q_with(&g, &gx, &qw, &gy, rows, path).unwrap();
                    let err = max_rel_err(&g.read(&gy).unwrap(), &want);
                    assert!(err <= REDUCING, "{kind:?} {path:?} {rows}x{k}x{n}: {err}");
                }
            }
        }
    }
}
