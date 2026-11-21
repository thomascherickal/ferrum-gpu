use super::*;

#[test]
fn sizes_eligibility_and_the_memory_ratio() {
    assert_eq!(QKind::Fp4.words() * 4, 20);
    assert_eq!(QKind::Fp8E4M3.words() * 4, 36);
    assert_eq!(QKind::Fp8E5M2.words() * 4, 36);
    assert_eq!(
        QKind::Q2K.words() * 4,
        88,
        "d, dmin, 16 scale bytes, 64 quant bytes"
    );
    assert_eq!((QKind::Fp4.elems(), QKind::Q2K.elems()), (32, 256));
    assert_eq!(packed_words(QKind::Fp4, 64, 3), Some(2 * 3 * 5));
    assert_eq!(packed_words(QKind::Fp8E4M3, 64, 3), Some(2 * 3 * 9));
    assert_eq!(packed_words(QKind::Q2K, 512, 3), Some(2 * 3 * 22));
    // Q2_K needs 256; the 32-element formats accept it too
    assert_eq!(packed_words(QKind::Q2K, 96, 3), None);
    assert!(eligible_for(QKind::Fp4, 96) && !eligible_for(QKind::Q2K, 96));
    assert!(eligible_for(QKind::Q2K, 768) && !eligible_for(QKind::Q2K, 0));
    assert_eq!(packed_words(QKind::Q2K, 256, usize::MAX), None, "overflow");
}

#[test]
fn only_the_formats_llama_cpp_has_are_ggml_types() {
    assert!(QKind::Fp8E4M3.ggml().is_none() && QKind::Fp8E5M2.ggml().is_none());
    for k in [QKind::Q8_0, QKind::Q4_0, QKind::F16, QKind::Fp4, QKind::Q2K] {
        assert!(k.ggml().is_some(), "{k:?}");
    }
}

#[test]
fn pack_puts_the_scale_first_and_keeps_the_quants() {
    // MXFP4: the scale byte becomes the f32 half scale; the 16 nibble bytes follow
    let mut b = vec![130u8];
    b.extend(0..16u8);
    let w = pack(QKind::Fp4, &b, 32, 1).unwrap();
    assert_eq!(w.len(), 5);
    assert_eq!(
        f32::from_bits(w[0]),
        2f32.powi(2),
        "e = 130 -> 2^(130 - 128)"
    );
    assert_eq!(w[1], u32::from_le_bytes([0, 1, 2, 3]));
    assert_eq!(w[4], u32::from_le_bytes([12, 13, 14, 15]));
    // FP8: like Q8_0, the f16 scale widened, then the 32 codes
    let mut b = f32_to_f16(0.25).to_le_bytes().to_vec();
    b.extend((0..32u8).map(|i| i * 3));
    let w = pack(QKind::Fp8E4M3, &b, 32, 1).unwrap();
    assert_eq!((w.len(), f32::from_bits(w[0])), (9, 0.25));
    assert_eq!(w[8], u32::from_le_bytes([84, 87, 90, 93]));
    // Q2_K: the file holds scales, quants, d, dmin; the device d, dmin, scales, quants
    let mut b: Vec<u8> = (0..80u8).collect();
    b.extend(f32_to_f16(1.5).to_le_bytes());
    b.extend(f32_to_f16(0.5).to_le_bytes());
    let w = pack(QKind::Q2K, &b, 256, 1).unwrap();
    assert_eq!(w.len(), 22);
    assert_eq!((f32::from_bits(w[0]), f32::from_bits(w[1])), (1.5, 0.5));
    assert_eq!(w[2], u32::from_le_bytes([0, 1, 2, 3]), "first scale word");
    assert_eq!(
        w[6],
        u32::from_le_bytes([16, 17, 18, 19]),
        "first quant word"
    );
    assert_eq!(w[21], u32::from_le_bytes([76, 77, 78, 79]));
}

#[test]
fn pack_and_quantise_refuse_bad_shapes() {
    for kind in EXT {
        let k = kind.elems();
        assert!(pack(kind, &[0; 100], k + 1, 1)
            .unwrap_err()
            .to_string()
            .contains("whole"));
        assert!(pack(kind, &[0; 3], k, 1)
            .unwrap_err()
            .to_string()
            .contains("needs"));
        assert!(quantise_matrix(kind, &vec![0.0; k + 1], k + 1, 1).is_err());
        assert!(quantise_matrix(kind, &vec![0.0; 2 * k - 1], k, 2).is_err());
    }
    // 32 divides 96 but 256 does not
    assert!(quantise_matrix(QKind::Q2K, &[0.0; 96], 96, 1).is_err());
    assert!(quantise_matrix(QKind::Fp4, &[0.0; 96], 96, 1).is_ok());
}

#[test]
fn requantising_dequantised_weights_stays_close() {
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
fn dequantised_values_match_the_cpu_decode() {
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
