use super::*;

#[test]
fn layout_covers_exactly_the_read_only_types() {
    for (ty, bytes, elems) in [
        (GGML_Q4_1, 20, 32),
        (GGML_Q5_0, 22, 32),
        (GGML_Q5_1, 24, 32),
        (GGML_Q8_1, 36, 32),
        (GGML_Q2_K, 84, 256),
        (GGML_Q3_K, 110, 256),
        (GGML_Q4_K, 144, 256),
        (GGML_Q5_K, 176, 256),
        (GGML_Q6_K, 210, 256),
    ] {
        assert_eq!(layout(ty), Some((bytes, elems)), "type {ty}");
        assert_eq!(dequant(ty, &vec![0u8; bytes * 2]).unwrap().len(), elems * 2);
    }
    assert_eq!(layout(crate::GGML_F32), None);
    assert_eq!(layout(crate::GGML_Q8_0), None);
    assert!(dequant(crate::GGML_Q4_0, &[0; 18]).is_none());
}

#[test]
fn q8_1_scales_the_signed_bytes_and_ignores_the_sum() {
    let mut raw = [f(0.5), f(7.0)].concat();
    raw.extend((0..32).map(|i| ((i - 16) as i8) as u8));
    for (i, v) in dequant(GGML_Q8_1, &raw).unwrap().iter().enumerate() {
        assert!(close(*v, 0.5 * (i as f32 - 16.0)));
    }
}

#[test]
fn q4_1_is_scale_times_nibble_plus_min() {
    let mut raw = [f(0.25), f(1.0)].concat();
    let los: Vec<u8> = (0..16).map(|j| (j % 15) as u8).collect();
    let his: Vec<u8> = (0..16).map(|j| ((j + 3) % 15) as u8).collect();
    raw.extend((0..16).map(|j| (his[j] << 4) | los[j]));
    let out = dequant(GGML_Q4_1, &raw).unwrap();
    for j in 0..16 {
        assert!(close(out[j], 0.25 * f32::from(los[j]) + 1.0));
        assert!(close(out[j + 16], 0.25 * f32::from(his[j]) + 1.0));
    }
}

#[test]
fn q5_takes_its_fifth_bit_from_qh() {
    let qh: u32 = 1 | (1 << 17); // element 0 and element 17
    let mut raw = f(2.0).to_vec();
    raw.extend(qh.to_le_bytes());
    raw.extend([0x21u8; 16]); // low nibble 1, high nibble 2
    let q = dequant(GGML_Q5_0, &raw).unwrap();
    assert_eq!(
        (q[0], q[1], q[17], q[16]),
        (2.0 * 1.0, 2.0 * -15.0, 2.0 * 2.0, 2.0 * -14.0)
    );
    raw.splice(2..2, f(0.5));
    let q = dequant(GGML_Q5_1, &raw).unwrap();
    assert_eq!(
        (q[0], q[17], q[1]),
        (2.0 * 17.0 + 0.5, 2.0 * 18.0 + 0.5, 2.0 * 1.0 + 0.5)
    );
}

#[test]
fn scale_min_unpacks_every_sub_block() {
    for j in 0..8 {
        assert_eq!(scale_min(j, &SCALES_D2_M0), (2, 0), "sub-block {j}");
    }
    // sub-blocks 4..8 combine the low nibble of byte j+4 with the top bits of byte j-4
    let q = [
        0b1100_0001,
        0,
        0,
        0,
        0b0011_0101,
        0,
        0,
        0,
        0b1111_0010,
        0,
        0,
        0,
    ];
    assert_eq!(scale_min(0, &q), (1, 0b11_0101));
    assert_eq!(
        scale_min(4, &q),
        (0b0010 | (0b11 << 4), 0b1111) // q[4] >> 6 is 0, so no high bits join the min
    );
}

#[test]
fn q4_k_constant_block() {
    // d * scale = 0.5 * 2 = 1, min 0, nibble 3 -> every value 3
    let mut raw = [f(0.5), f(0.0)].concat();
    raw.extend(SCALES_D2_M0);
    raw.extend([0x33u8; 128]);
    assert!(dequant(GGML_Q4_K, &raw)
        .unwrap()
        .iter()
        .all(|&v| close(v, 3.0)));
}

#[test]
fn q4_k_applies_the_min_and_separates_low_from_high_nibbles() {
    // scale 2 and min 1 everywhere: low nibbles 5 -> 1*5 - 0.5*1... use d=1, dmin=0.5: 2*5 - 0.5
    let scales = [2u8, 2, 2, 2, 1, 1, 1, 1, 0x12, 0x12, 0x12, 0x12];
    let mut raw = [f(1.0), f(0.5)].concat();
    raw.extend(scales);
    raw.extend([0x75u8; 128]); // low 5, high 7
    let out = dequant(GGML_Q4_K, &raw).unwrap();
    // pair 0: sub-blocks 0 (scale 2, min 1) and 1 (scale 2, min 1)
    assert!(close(out[0], 2.0 * 5.0 - 0.5 * 1.0) && close(out[32], 2.0 * 7.0 - 0.5 * 1.0));
    // pair 2: sub-blocks 4 and 5 use the packed form: scale = 0x2 | (0 << 4) = 2, min = 0x1
    assert!(close(out[128], 2.0 * 5.0 - 0.5 * 1.0) && close(out[160], 2.0 * 7.0 - 0.5 * 1.0));
}

#[test]
fn q5_k_low_and_high_bits() {
    let build = |qh: u8| {
        let mut raw = [f(0.5), f(0.0)].concat();
        raw.extend(SCALES_D2_M0);
        raw.extend([qh; 32]);
        raw.extend([0x33u8; 128]);
        raw
    };
    assert!(dequant(GGML_Q5_K, &build(0))
        .unwrap()
        .iter()
        .all(|&v| close(v, 3.0)));
    assert!(dequant(GGML_Q5_K, &build(0xff))
        .unwrap()
        .iter()
        .all(|&v| close(v, 19.0)));
}

#[test]
fn q6_k_constant_block() {
    // ql nibbles 8, qh fields 2 -> q = (8 | 2 << 4) - 32 = 8; scales 2, d 0.5 -> 8
    let mut raw = vec![0x88u8; 128];
    raw.extend([0xAAu8; 64]);
    raw.extend(std::iter::repeat_n(2i8 as u8, 16));
    raw.extend(f(0.5));
    assert!(dequant(GGML_Q6_K, &raw)
        .unwrap()
        .iter()
        .all(|&v| close(v, 8.0)));
}
