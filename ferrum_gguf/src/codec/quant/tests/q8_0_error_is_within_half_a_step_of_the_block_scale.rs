use super::*;

#[test]
fn q8_0_error_is_within_half_a_step_of_the_block_scale() {
    let d = data(96);
    let back = decode(GGML_Q8_0, &encode(GGML_Q8_0, &d).unwrap(), 96).unwrap();
    for (blk, got) in d.chunks(QK).zip(back.chunks(QK)) {
        let step = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs())) / 127.0;
        for (a, b) in blk.iter().zip(got) {
            assert!(
                (a - b).abs() <= step * 0.51 + 1e-4,
                "{a} vs {b} step {step}"
            );
        }
    }
}

#[test]
fn q8_0_known_block() {
    let mut blk = vec![0.0f32; 32];
    blk[0] = 127.0;
    blk[1] = -63.5;
    let bytes = encode(GGML_Q8_0, &blk).unwrap();
    assert_eq!(f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])), 1.0);
    assert_eq!(bytes[2] as i8, 127);
    assert_eq!(bytes[3] as i8, -64); // -63.5 rounds away from zero
    assert_eq!(decode(GGML_Q8_0, &bytes, 32).unwrap()[0], 127.0);
}

#[test]
fn q4_0_is_coarse_but_bounded_and_idempotent() {
    let d = data(64);
    let once = decode(GGML_Q4_0, &encode(GGML_Q4_0, &d).unwrap(), 64).unwrap();
    for (blk, got) in d.chunks(QK).zip(once.chunks(QK)) {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        for (a, b) in blk.iter().zip(got) {
            assert!((a - b).abs() <= amax / 8.0 * 1.01 + 1e-4, "{a} vs {b}");
        }
    }
    let twice = decode(GGML_Q4_0, &encode(GGML_Q4_0, &once).unwrap(), 64).unwrap();
    assert_eq!(once, twice, "requantising decoded values is stable");
}

#[test]
fn q4_0_nibble_order_is_low_half_first() {
    let mut blk = vec![0.0f32; 32];
    blk[0] = -8.0; // largest magnitude, becomes q=0 -> -8*d with d = 1
    blk[16] = 7.0;
    let back = decode(GGML_Q4_0, &encode(GGML_Q4_0, &blk).unwrap(), 32).unwrap();
    assert_eq!(back[0], -8.0);
    assert_eq!(back[16], 7.0);
    assert_eq!(back[1], 0.0);
}

#[test]
fn zero_blocks_and_length_mismatches() {
    assert_eq!(
        decode(GGML_Q8_0, &encode(GGML_Q8_0, &[0.0; 32]).unwrap(), 32).unwrap(),
        vec![0.0; 32]
    );
    assert_eq!(
        decode(GGML_Q4_0, &encode(GGML_Q4_0, &[0.0; 32]).unwrap(), 32).unwrap(),
        vec![0.0; 32]
    );
    assert!(decode(GGML_F32, &[0; 7], 2).is_err());
    assert!(encode(GGML_Q8_0, &[1.0; 5]).is_err());
}
