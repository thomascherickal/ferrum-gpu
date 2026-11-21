use super::*;

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
    assert_eq!(QKind::F16.words() * 4, 64, "32 f16 values and no scale");
    assert_eq!(packed_words(QKind::F16, 64, 3), Some(2 * 3 * 16));
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
