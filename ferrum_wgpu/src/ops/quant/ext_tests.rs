//! The weight formats added after Q8_0, Q4_0 and F16: MXFP4, FP8 (E4M3, E5M2) and Q2_K.

use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, REDUCING};
use ferrum_gguf::f16::f32_to_f16;
use ferrum_ref::matmul::{matmul, rand_vec};

pub(super) const EXT: [QKind; 4] = [QKind::Fp4, QKind::Fp8E4M3, QKind::Fp8E5M2, QKind::Q2K];

/// Shapes `(k, n)` that fill whole blocks of `kind`, including awkward widths for the kernels.
fn shapes(kind: QKind) -> Vec<(usize, usize)> {
    if kind == QKind::Q2K {
        vec![(256, 1), (256, 5), (512, 17)]
    } else {
        vec![(32, 1), (32, 5), (64, 3), (96, 65)]
    }
}

fn weights(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = rand_vec(n, 21).into_iter().map(|x| x * 2.0).collect();
    for (i, x) in v.iter_mut().enumerate() {
        match i / 32 % 5 {
            0 => *x *= 1e-5,
            1 => *x = 0.0,
            2 => *x *= 300.0,
            _ => {}
        }
    }
    v
}

fn rows_of(kind: QKind, bytes: &[u8], k: usize, n: usize) -> Vec<f32> {
    dequantise_bytes(kind, bytes, k * n).unwrap()
}

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

#[test]
fn linear_matches_the_cpu_matmul_on_the_dequantised_weights() {
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
fn the_packed_weight_is_smaller_by_the_claimed_ratio() {
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
fn every_kernel_matches_the_cpu_matmul_for_the_extra_formats() {
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

#[test]
fn the_kernel_choice_for_the_extra_formats_follows_the_measurements() {
    // several rows: tiled
    for kind in EXT {
        assert_eq!(choose(32, kind, 512, 1408), QPath::Tiled);
        assert_eq!(choose(2, kind, 256, 3), QPath::Tiled);
    }
    // one row: per-column unless the output is wider than the input, where MXFP4 and FP8 are
    // faster naive on the MX330; Q2_K always per-column
    for kind in [QKind::Fp4, QKind::Fp8E4M3, QKind::Fp8E5M2] {
        assert_eq!(choose(1, kind, 512, 512), QPath::Gemv);
        assert_eq!(choose(1, kind, 1024, 512), QPath::Gemv);
        assert_eq!(choose(1, kind, 512, 1408), QPath::Naive);
    }
    for (k, n) in [(512, 512), (1024, 512), (512, 1408), (512, 2048)] {
        assert_eq!(choose(1, QKind::Q2K, k, n), QPath::Gemv);
    }
}

#[test]
fn the_per_column_kernel_still_refuses_more_than_one_row_for_the_extra_formats() {
    let Some(g) = open_for_test() else { return };
    let qw = upload_quantised(&g, QKind::Fp4, &rand_vec(32 * 4, 2), 32, 4).unwrap();
    let (gx, gy) = (g.upload(&rand_vec(64, 3)).unwrap(), g.zeros(8).unwrap());
    assert!(linear_q_with(&g, &gx, &qw, &gy, 2, QPath::Gemv).is_err());
}

#[test]
fn the_quantised_product_stays_near_the_f32_product() {
    let Some(g) = open_for_test() else { return };
    // loose bounds that only fail if a format is decoded wrongly: 2-bit values are coarse
    for (kind, k, bound) in [
        (QKind::Fp4, 128, 0.35),
        (QKind::Fp8E4M3, 128, 0.04),
        (QKind::Fp8E5M2, 128, 0.08),
        (QKind::Q2K, 256, 0.9),
    ] {
        let (n, rows) = (24, 5);
        let w = rand_vec(k * n, 4);
        let x = rand_vec(rows * k, 6);
        let want = matmul(&x, &w, rows, k, n).unwrap();
        let qw = upload_quantised(&g, kind, &w, k, n).unwrap();
        let (gx, gy) = (g.upload(&x).unwrap(), g.zeros(rows * n).unwrap());
        linear_q(&g, &gx, &qw, &gy, rows).unwrap();
        let got = g.read(&gy).unwrap();
        let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(worst <= bound * scale, "{kind:?}: {worst} vs {scale}");
    }
}
