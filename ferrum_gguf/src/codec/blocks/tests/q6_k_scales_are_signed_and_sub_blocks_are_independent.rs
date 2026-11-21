use super::*;

#[test]
fn q6_k_scales_are_signed_and_sub_blocks_are_independent() {
    let mut raw = vec![0x88u8; 128];
    raw.extend([0xAAu8; 64]);
    raw.extend((0..16).map(|i| (i as i8 - 8) as u8)); // scales -8..7
    raw.extend(f(1.0));
    let out = dequant(GGML_Q6_K, &raw).unwrap();
    // elements 0..16 of the first half use scale index 0, 16..32 index 1
    assert!(close(out[0], -8.0 * 8.0) && close(out[16], -7.0 * 8.0));
    assert!(close(out[32], -6.0 * 8.0) && close(out[128], 0.0 * 8.0));
}

#[test]
fn q2_k_scales_and_mins() {
    // every 2-bit q = 1, d = 1, dmin = 0.5; scales[i] = i | (15 - i) << 4 -> y = i - 0.5 * (15 - i)
    let mut b = vec![0u8; Q2_K_BLOCK];
    for (i, s) in b[..16].iter_mut().enumerate() {
        *s = (i as u8) | (((15 - i) as u8) << 4);
    }
    b[16..80].fill(0x55);
    b[80..82].copy_from_slice(&f(1.0));
    b[82..84].copy_from_slice(&f(0.5));
    for (g, chunk) in dequant(GGML_Q2_K, &b).unwrap().chunks_exact(16).enumerate() {
        let want = g as f32 - 0.5 * (15 - g) as f32;
        assert!(chunk.iter().all(|&v| close(v, want)), "group {g}");
    }
}

#[test]
fn q3_k_high_bit_offsets() {
    // low bits 1, all scales 34 -> dl = 2; hmask set -> q = 1 -> 2; hmask clear -> q = -3 -> -6
    let mut b = vec![0u8; Q3_K_BLOCK];
    b[32..96].fill(0x55);
    b[96..104].fill(0x22);
    b[104..108].fill(0xAA);
    b[108..110].copy_from_slice(&f(1.0));
    b[..32].fill(0xff);
    assert!(dequant(GGML_Q3_K, &b)
        .unwrap()
        .iter()
        .all(|&v| close(v, 2.0)));
    b[..32].fill(0);
    assert!(dequant(GGML_Q3_K, &b)
        .unwrap()
        .iter()
        .all(|&v| close(v, -6.0)));
}

#[test]
fn whole_blocks_only_and_the_tail_is_ignored() {
    assert_eq!(dequant(GGML_Q4_1, &[0u8; 20 * 3 + 7]).unwrap().len(), 96);
    assert!(dequant(GGML_Q6_K, &[]).unwrap().is_empty());
}
