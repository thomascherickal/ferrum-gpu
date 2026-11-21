use super::*;

#[test]
fn dequantised_values_are_bit_identical_to_the_cpu_decode() {
    let Some(g) = open_for_test() else { return };
    for kind in KINDS {
        for (k, n) in [(32usize, 1usize), (32, 5), (64, 3), (96, 65), (128, 64)] {
            let bytes = quantise_matrix(kind, &nasty(k * n), k, n).unwrap();
            let rows = decoded_rows(kind, &bytes, k, n); // rows[col * k + i]
            let w = upload_qweight(&g, kind, &bytes, k, n).unwrap();
            // x = identity reads the weight back: y[r, col] = W[col, r]
            let mut eye = vec![0.0f32; k * k];
            (0..k).for_each(|i| eye[i * k + i] = 1.0);
            let (gx, gy) = (g.upload(&eye).unwrap(), g.zeros(k * n).unwrap());
            linear_q(&g, &gx, &w, &gy, k).unwrap();
            let got = g.read(&gy).unwrap();
            for r in 0..k {
                for col in 0..n {
                    assert_eq!(
                        got[r * n + col],
                        rows[col * k + r],
                        "{kind:?} {k}x{n} element [{col}][{r}]"
                    );
                }
            }
        }
    }
}

#[test]
fn linear_matches_the_cpu_matmul_on_the_dequantised_weights() {
    let Some(g) = open_for_test() else { return };
    for kind in KINDS {
        for (rows, k, n) in [
            (1usize, 32usize, 1usize),
            (1, 256, 100),
            (7, 96, 13),
            (33, 64, 65),
            (2, 512, 3),
        ] {
            let w = rand_vec(k * n, 4);
            let bytes = quantise_matrix(kind, &w, k, n).unwrap();
            let want_w = transpose(&decoded_rows(kind, &bytes, k, n), n, k); // [k, n]
            let x = rand_vec(rows * k, 5);
            let want = matmul(&x, &want_w, rows, k, n).unwrap();
            let qw = upload_qweight(&g, kind, &bytes, k, n).unwrap();
            let (gx, gy) = (g.upload(&x).unwrap(), g.zeros(rows * n).unwrap());
            linear_q(&g, &gx, &qw, &gy, rows).unwrap();
            let e = max_rel_err(&g.read(&gy).unwrap(), &want);
            eprintln!("linear_q {kind:?} {rows}x{k}x{n}: max rel err {e:.2e} (tol {REDUCING:.0e})");
            assert!(e < REDUCING, "{kind:?} {rows}x{k}x{n}: {e}");
        }
    }
}

#[test]
fn quantised_output_stays_close_to_the_f32_product() {
    // Quantisation noise on a 256-term sum of uniform weights: Q8_0 about 0.02 typical, Q4_0 several
    // times that. The bounds are sanity limits, not calibrated tolerances (dequantisation itself is
    // proven bit-exact above).
    let Some(g) = open_for_test() else { return };
    let (rows, k, n) = (4, 256, 64);
    let (w, x) = (rand_vec(k * n, 7), rand_vec(rows * k, 8));
    let exact = matmul(&x, &w, rows, k, n).unwrap();
    let err = |kind| {
        let qw = upload_quantised(&g, kind, &w, k, n).unwrap();
        let (gx, gy) = (g.upload(&x).unwrap(), g.zeros(rows * n).unwrap());
        linear_q(&g, &gx, &qw, &gy, rows).unwrap();
        max_rel_err(&g.read(&gy).unwrap(), &exact)
    };
    let (e8, e4, e16) = (err(QKind::Q8_0), err(QKind::Q4_0), err(QKind::F16));
    eprintln!("quantised vs f32 product: q8 {e8:.2e}, q4 {e4:.2e}, f16 {e16:.2e}");
    assert!(e8 < 0.15 && e4 < 2.0 && e8 * 5.0 < e4);
    // f16 keeps 11 significant bits: about 5e-4 per weight, so a 256-term sum is far closer than Q8_0
    assert!(e16 < 5e-3 && e16 * 10.0 < e8, "f16 {e16} vs q8 {e8}");
}
