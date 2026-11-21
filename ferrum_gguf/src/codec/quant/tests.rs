#[test]
fn read_only_block_types_have_sizes_and_decode_but_cannot_be_encoded() {
    use crate::{GGML_Q4_1, GGML_Q4_K, GGML_Q5_0, GGML_Q5_1, GGML_Q6_K, GGML_Q8_1};
    for (ty, bytes, elems) in [
        (GGML_Q4_1, 20, 32),
        (GGML_Q5_0, 22, 32),
        (GGML_Q5_1, 24, 32),
        (GGML_Q8_1, 36, 32),
        (GGML_Q4_K, 144, 256),
        (GGML_Q6_K, 210, 256),
    ] {
        assert_eq!(nbytes(ty, elems * 3).unwrap(), bytes * 3);
        assert!(nbytes(ty, elems + 1).unwrap_err().0.contains("multiple"));
        assert_eq!(decode(ty, &vec![0u8; bytes], elems).unwrap().len(), elems);
        assert!(
            decode(ty, &vec![0u8; bytes + 1], elems).is_err(),
            "length is checked"
        );
        assert!(encode(ty, &vec![0.0; elems]).is_err());
    }
}

#[test]
fn q2_k_and_mxfp4_have_sizes_and_can_be_written_and_read_back() {
    for (ty, bytes, elems) in [(GGML_Q2_K, 84, 256), (GGML_MXFP4, 17, 32)] {
        assert_eq!(nbytes(ty, elems * 3).unwrap(), bytes * 3);
        assert!(nbytes(ty, elems + 1).unwrap_err().0.contains("multiple"));
        let data = (0..elems * 2)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect::<Vec<_>>();
        let enc = encode(ty, &data).unwrap();
        assert_eq!(enc.len(), bytes * 2);
        assert_eq!(decode(ty, &enc, data.len()).unwrap().len(), data.len());
        assert!(encode(ty, &data[..elems + 1]).is_err(), "whole blocks only");
    }
}

#[test]
fn integer_tensor_types_have_sizes_but_no_f32_form() {
    assert_eq!(nbytes(crate::GGML_I16, 5).unwrap(), 10);
    assert_eq!(nbytes(crate::GGML_I32, 5).unwrap(), 20);
    assert!(decode(crate::GGML_I16, &[0; 10], 5).is_err());
    assert!(decode(crate::GGML_I32, &[0; 20], 5).is_err());
}

use super::*;

fn data(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
        .collect()
}

#[test]
fn sizes() {
    assert_eq!(nbytes(GGML_F32, 5).unwrap(), 20);
    assert_eq!(nbytes(GGML_F16, 5).unwrap(), 10);
    assert_eq!(nbytes(GGML_Q8_0, 64).unwrap(), 68);
    assert_eq!(nbytes(GGML_Q4_0, 64).unwrap(), 36);
    assert!(nbytes(GGML_Q8_0, 33).is_err());
    assert!(nbytes(GGML_Q4_0, 31).is_err());
    assert!(nbytes(GGML_F32, usize::MAX).is_err());
    assert!(nbytes(15, 256) // Q8_K, not supported
        .unwrap_err()
        .to_string()
        .contains("unsupported"));
}

#[test]
fn f32_and_f16_round_trip() {
    let d = data(70);
    assert_eq!(
        decode(GGML_F32, &encode(GGML_F32, &d).unwrap(), 70).unwrap(),
        d
    );
    let back = decode(GGML_F16, &encode(GGML_F16, &d).unwrap(), 70).unwrap();
    assert!(d
        .iter()
        .zip(&back)
        .all(|(a, b)| (a - b).abs() <= a.abs() * 1e-3 + 1e-6));
}

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
