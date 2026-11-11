// The hand-computed cases below follow the unit tests in ferrum-cpu's dequant.rs (commit
// dbf6392c8883f293ac64b28a29c55e823179b257): constant blocks whose answer follows from the format.
use super::*;
use crate::f16::f32_to_f16;

fn f(v: f32) -> [u8; 2] {
    f32_to_f16(v).to_le_bytes()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

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

/// 12 packed bytes that unpack to scale 2, min 0 for all eight sub-blocks.
const SCALES_D2_M0: [u8; 12] = [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2];

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
