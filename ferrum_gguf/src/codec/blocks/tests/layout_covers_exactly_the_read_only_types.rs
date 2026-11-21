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
