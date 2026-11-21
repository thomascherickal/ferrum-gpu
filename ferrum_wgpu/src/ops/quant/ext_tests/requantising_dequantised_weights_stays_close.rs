use super::*;

#[test]
pub(super) fn requantising_dequantised_weights_stays_close() {
    // MXFP4 values are on a grid whose requantisation is exact; FP8 and Q2_K may move a value by
    // one step of their grid when the block scale is re-derived
    for kind in EXT {
        for (k, n) in shapes(kind) {
            let bytes = quantise_matrix(kind, &weights(k * n), k, n).unwrap();
            let rows = rows_of(kind, &bytes, k, n);
            let again = quantise_matrix(kind, &transpose(&rows, n, k), k, n).unwrap();
            let back = rows_of(kind, &again, k, n);
            if kind == QKind::Fp4 {
                assert_eq!(back, rows, "MXFP4 {k}x{n}");
            } else {
                let amax = rows.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let worst = rows
                    .iter()
                    .zip(&back)
                    .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
                assert!(
                    worst <= 0.35 * amax + 1e-30,
                    "{kind:?} {k}x{n}: {worst} vs {amax}"
                );
            }
        }
    }
}

#[test]
pub(super) fn dequantised_values_match_the_cpu_decode() {
    let Some(g) = open_for_test() else { return };
    for kind in EXT {
        for (k, n) in shapes(kind) {
            let bytes = quantise_matrix(kind, &weights(k * n), k, n).unwrap();
            let rows = rows_of(kind, &bytes, k, n);
            let w = upload_qweight(&g, kind, &bytes, k, n).unwrap();
            let mut eye = vec![0.0f32; k * k];
            (0..k).for_each(|i| eye[i * k + i] = 1.0);
            let (gx, gy) = (g.upload(&eye).unwrap(), g.zeros(k * n).unwrap());
            linear_q(&g, &gx, &w, &gy, k).unwrap();
            let got = g.read(&gy).unwrap();
            for r in 0..k {
                for col in 0..n {
                    let (a, b) = (got[r * n + col], rows[col * k + r]);
                    if kind == QKind::Q2K {
                        // d * q - m may be one fused operation on the device: one rounding apart
                        assert!(
                            (a - b).abs() <= 2e-6 * b.abs().max(1e-30),
                            "{kind:?} [{col}][{r}] {a} {b}"
                        );
                    } else {
                        assert_eq!(a, b, "{kind:?} {k}x{n} element [{col}][{r}]");
                    }
                }
            }
        }
    }
}
