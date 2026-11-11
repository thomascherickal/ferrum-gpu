//! Tensor element encodings: sizes, decode to f32 and encode from f32 (F32, F16, Q8_0, Q4_0).

use crate::f16::{f16_to_f32, f32_to_f16};
use crate::reader::GgufError;
use crate::{GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q8_0, QK};

fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

/// Bytes a tensor of `n` elements occupies on disk, or an error for unsupported types or
/// element counts that do not fill whole blocks.
pub fn nbytes(ty: u32, n: usize) -> Result<usize, GgufError> {
    let blocks = |per: usize| {
        if !n.is_multiple_of(QK) {
            return Err(err(format!(
                "{n} elements is not a multiple of the block size {QK}"
            )));
        }
        (n / QK)
            .checked_mul(per)
            .ok_or_else(|| err("tensor size overflows"))
    };
    match ty {
        GGML_F32 => n.checked_mul(4).ok_or_else(|| err("tensor size overflows")),
        GGML_F16 => n.checked_mul(2).ok_or_else(|| err("tensor size overflows")),
        GGML_Q8_0 => blocks(2 + QK),
        GGML_Q4_0 => blocks(2 + QK / 2),
        other => Err(err(format!("unsupported ggml tensor type {other}"))),
    }
}

/// Decode `n` elements of type `ty` from `bytes` (which must be exactly [`nbytes`] long).
pub fn decode(ty: u32, bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if bytes.len() != nbytes(ty, n)? {
        return Err(err(
            "tensor byte length does not match its type and element count",
        ));
    }
    let h = |b: &[u8]| f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    Ok(match ty {
        GGML_F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        GGML_F16 => bytes.chunks_exact(2).map(h).collect(),
        GGML_Q8_0 => bytes
            .chunks_exact(2 + QK)
            .flat_map(|b| {
                let d = h(b);
                b[2..].iter().map(move |&q| d * f32::from(q as i8))
            })
            .collect(),
        _ => bytes
            .chunks_exact(2 + QK / 2) // Q4_0: low nibbles are elements 0..16, high nibbles 16..32
            .flat_map(|b| {
                let d = h(b);
                let lo = b[2..].iter().map(move |&q| d * (f32::from(q & 0x0f) - 8.0));
                let hi = b[2..].iter().map(move |&q| d * (f32::from(q >> 4) - 8.0));
                lo.chain(hi).collect::<Vec<_>>()
            })
            .collect(),
    })
}

/// Encode f32 data as type `ty` (round-half-away for Q8_0, like llama.cpp's reference quantiser).
pub fn encode(ty: u32, data: &[f32]) -> Result<Vec<u8>, GgufError> {
    let mut out = Vec::with_capacity(nbytes(ty, data.len())?);
    match ty {
        GGML_F32 => data
            .iter()
            .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        GGML_F16 => data
            .iter()
            .for_each(|&v| out.extend_from_slice(&f32_to_f16(v).to_le_bytes())),
        GGML_Q8_0 => {
            for blk in data.chunks_exact(QK) {
                let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
                out.extend(blk.iter().map(|&v| (v * id).round() as i8 as u8));
            }
        }
        _ => {
            for blk in data.chunks_exact(QK) {
                // The element with the largest magnitude (keeping its sign) maps to -8.
                let max = blk
                    .iter()
                    .copied()
                    .fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
                let d = max / -8.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
                let q = |v: f32| ((v * id + 8.5) as i32).clamp(0, 15) as u8;
                out.extend((0..QK / 2).map(|j| q(blk[j]) | (q(blk[j + QK / 2]) << 4)));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
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
        assert!(nbytes(14, 256)
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
}
