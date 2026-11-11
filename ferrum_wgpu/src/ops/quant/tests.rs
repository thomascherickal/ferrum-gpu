use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, REDUCING};
use ferrum_gguf::quant::decode;
use ferrum_ref::matmul::{matmul, rand_vec};

const KINDS: [QKind; 2] = [QKind::Q8_0, QKind::Q4_0];

/// Values that stress the quantisers: spread magnitudes, all-zero blocks, tiny blocks (the f16
/// scale is subnormal), large blocks and exact small integers.
fn nasty(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = rand_vec(n, 9).into_iter().map(|x| x * 3.0).collect();
    for (i, x) in v.iter_mut().enumerate() {
        match i / 32 % 7 {
            0 => *x *= 1e-6,
            1 => *x = 0.0,
            2 => *x *= 1e4,
            3 => *x = (i % 9) as f32 - 4.0,
            _ => {}
        }
    }
    v
}

/// What the CPU sees for a quantised `[k, n]` weight: decode of the GGUF bytes, `[n, k]` rows.
fn decoded_rows(kind: QKind, bytes: &[u8], k: usize, n: usize) -> Vec<f32> {
    decode(kind.ggml(), bytes, k * n).unwrap()
}

#[test]
fn sizes_and_eligibility() {
    assert!(eligible(32) && eligible(96) && !eligible(0) && !eligible(33));
    assert_eq!(packed_words(QKind::Q8_0, 64, 3), Some(2 * 3 * 9));
    assert_eq!(packed_words(QKind::Q4_0, 64, 3), Some(2 * 3 * 5));
    assert_eq!(packed_words(QKind::Q8_0, 33, 3), None);
    assert_eq!(packed_words(QKind::Q8_0, 32, usize::MAX), None, "overflow");
    // the headline claim: 128 f32 bytes per block column against 36 and 20 on the device
    assert_eq!(QKind::Q8_0.words() * 4, 36);
    assert_eq!(QKind::Q4_0.words() * 4, 20);
}

#[test]
fn pack_reorders_blocks_column_major_and_widens_the_scale() {
    // k = 64 (2 blocks), n = 2: file order is (col0 b0, col0 b1, col1 b0, col1 b1).
    let (k, n) = (64, 2);
    let mut bytes = Vec::new();
    for blk in 0..4u8 {
        let d = ferrum_gguf::f16::f32_to_f16(f32::from(blk) + 1.0);
        bytes.extend(d.to_le_bytes());
        bytes.extend((0..32u8).map(|i| i.wrapping_add(blk * 40)));
    }
    let w = pack(QKind::Q8_0, &bytes, k, n).unwrap();
    assert_eq!(w.len(), 2 * 2 * 9);
    // record (b, col) lives at (b * n + col) * 9
    for (b, col, file_blk) in [(0, 0, 0u8), (1, 0, 1), (0, 1, 2), (1, 1, 3)] {
        let rec = &w[(b * n + col) * 9..][..9];
        assert_eq!(f32::from_bits(rec[0]), f32::from(file_blk) + 1.0);
        let first = u32::from_le_bytes([0, 1, 2, 3].map(|i: u8| i.wrapping_add(file_blk * 40)));
        assert_eq!(rec[1], first, "block {file_blk} quants");
    }
    // Q4_0: 16 quant bytes become 4 words after the scale
    let mut q4 = Vec::new();
    q4.extend(ferrum_gguf::f16::f32_to_f16(0.5).to_le_bytes());
    q4.extend(0..16u8);
    let w = pack(QKind::Q4_0, &q4, 32, 1).unwrap();
    assert_eq!(w.len(), 5);
    assert_eq!(f32::from_bits(w[0]), 0.5);
    assert_eq!(w[4], u32::from_le_bytes([12, 13, 14, 15]));
}

#[test]
fn pack_and_quantise_refuse_bad_shapes() {
    for kind in KINDS {
        assert!(pack(kind, &[0; 100], 33, 1)
            .unwrap_err()
            .to_string()
            .contains("whole"));
        assert!(pack(kind, &[0; 3], 32, 1)
            .unwrap_err()
            .to_string()
            .contains("needs"));
        assert!(quantise_matrix(kind, &[0.0; 33], 33, 1).is_err());
        assert!(quantise_matrix(kind, &[0.0; 63], 32, 2).is_err());
    }
}

#[test]
fn requantising_dequantised_weights_reproduces_the_values() {
    // GpuLlama quantises the f32 weights of a model that was loaded from a quantised GGUF; that
    // is only the same model if decode(encode(decode(bytes))) == decode(bytes), bit for bit. (The
    // bytes themselves can differ: for a block so small that its f16 scale underflows to zero the
    // encoder keeps nonzero quants, and the block decodes to zeros either way.)
    for kind in KINDS {
        for (k, n) in [(32usize, 1usize), (64, 7), (96, 40), (256, 9)] {
            let w = nasty(k * n);
            let bytes = quantise_matrix(kind, &w, k, n).unwrap();
            let rows = decoded_rows(kind, &bytes, k, n); // [n, k]
            let again = quantise_matrix(kind, &transpose(&rows, n, k), k, n).unwrap();
            assert_eq!(decoded_rows(kind, &again, k, n), rows, "{kind:?} {k}x{n}");
        }
    }
}

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
    let (e8, e4) = (err(QKind::Q8_0), err(QKind::Q4_0));
    eprintln!("quantised vs f32 product: q8 {e8:.2e}, q4 {e4:.2e}");
    assert!(e8 < 0.15 && e4 < 2.0 && e8 * 5.0 < e4);
}

#[test]
fn the_packed_weight_is_smaller_by_the_claimed_ratio() {
    let Some(g) = open_for_test() else { return };
    let (k, n) = (64, 32);
    for (kind, ratio) in [(QKind::Q8_0, 128.0 / 36.0), (QKind::Q4_0, 128.0 / 20.0)] {
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
fn linear_refuses_mismatched_buffers_and_fails_cleanly() {
    let Some(g) = open_for_test() else { return };
    let qw = upload_quantised(&g, QKind::Q8_0, &rand_vec(64 * 4, 1), 64, 4).unwrap();
    let (x, y) = (g.zeros(2 * 64).unwrap(), g.zeros(2 * 4).unwrap());
    assert!(linear_q(&g, &x, &qw, &y, 3)
        .unwrap_err()
        .to_string()
        .contains("linear_q x"));
    assert!(linear_q(&g, &x, &qw, &g.zeros(7).unwrap(), 2)
        .unwrap_err()
        .to_string()
        .contains("linear_q y"));
    assert_eq!(g.sweep_faults(|| linear_q(&g, &x, &qw, &y, 2)), 1);
    assert!(upload_quantised(&g, QKind::Q4_0, &[0.0; 33], 33, 1).is_err());
}
