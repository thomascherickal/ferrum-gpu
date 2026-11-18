//! MXFP4 (ggml type 39): blocks of 32 values, each a 4-bit E2M1 float, sharing one E8M0 scale
//! (a power of two). 17 bytes per block: the scale byte, then 16 bytes of nibbles (low nibbles are
//! elements 0..16, high nibbles 16..32). llama.cpp reads this type, so a file we write runs there.
//!
//! The stored nibble indexes `VALUES`, which are twice the E2M1 values (0, 0.5, 1, 1.5, 2, 3, 4,
//! 6) so they are integers; the scale byte `e` means `2^(e - 127)`, and the decoder applies half
//! of it.

use crate::QK;

/// Bytes of one block.
pub const BLOCK: usize = 1 + QK / 2;

/// The sixteen codes as twice their E2M1 value: sign bit is the top bit of the nibble.
pub const VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// The largest scale byte used (255 is NaN in E8M0).
const MAX_E: i32 = 254;

/// `2^(e - 128)`: half of the E8M0 scale `e`, exact for every byte (the two smallest are
/// subnormal f32s).
pub fn half_scale(e: u8) -> f32 {
    if e < 2 {
        f32::from_bits(0x0020_0000u32 << e)
    } else {
        f32::from_bits((u32::from(e) - 1) << 23)
    }
}

/// Decode whole blocks to f32.
pub fn decode(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK * QK);
    for b in bytes.chunks_exact(BLOCK) {
        let d = half_scale(b[0]);
        let q = &b[1..];
        out.extend(
            q.iter()
                .map(|&x| f32::from(VALUES[usize::from(x & 15)]) * d),
        );
        out.extend(
            q.iter()
                .map(|&x| f32::from(VALUES[usize::from(x >> 4)]) * d),
        );
    }
    out
}

/// The scale byte for a block whose largest magnitude is `amax`: the power of two that puts the
/// largest value at the top of the E2M1 range (exponent 2), as the reference quantiser does.
fn scale_byte(amax: f32) -> u8 {
    if amax <= 0.0 || !amax.is_finite() {
        return 0;
    }
    ((amax.log2().floor() as i32 - 2 + 127).clamp(0, MAX_E)) as u8
}

/// The code (index into [`VALUES`]) nearest to `x` at the scale `d` (half scale); a tie goes to the
/// lower code.
fn nearest(x: f32, d: f32) -> u8 {
    let mut best = 0usize;
    let mut best_err = (f32::from(VALUES[0]) * d - x).abs();
    for (i, &v) in VALUES.iter().enumerate().skip(1) {
        let err = (f32::from(v) * d - x).abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best as u8
}

/// Encode whole blocks of 32 values; the length must be a multiple of 32.
pub fn encode(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / QK * BLOCK);
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let e = scale_byte(amax);
        let d = half_scale(e);
        out.push(e);
        out.extend((0..QK / 2).map(|j| nearest(blk[j], d) | (nearest(blk[j + QK / 2], d) << 4)));
    }
    out
}

#[cfg(test)]
mod tests;
