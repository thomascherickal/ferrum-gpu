//! 8-bit floats for GPU weight storage: E4M3 (the OCP "fn" variant: bias 7, no infinities, the
//! all-ones code is NaN, largest value 448) and E5M2 (IEEE-like: bias 15, infinities and NaN,
//! largest finite value 57344). llama.cpp has no FP8 tensor type, so these never go into a GGUF
//! file; `ferrum-gpu` converts a model to them when it loads it onto the device.
//!
//! A block is 32 values with an f16 scale (the largest magnitude divided by the format's largest
//! value) and one code byte each: 34 bytes, laid out like Q8_0 so one packer serves both.

use crate::f16::{f16_to_f32, f32_to_f16};
use crate::QK;

/// Bytes of one block.
pub const BLOCK: usize = 2 + QK;

/// The two formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8 {
    /// 4 exponent bits, 3 mantissa bits, bias 7, largest value 448.
    E4M3,
    /// 5 exponent bits, 2 mantissa bits, bias 15, largest finite value 57344.
    E5M2,
}

impl Fp8 {
    /// The largest finite value.
    pub fn max(self) -> f32 {
        match self {
            Fp8::E4M3 => 448.0,
            Fp8::E5M2 => 57344.0,
        }
    }

    /// The value of a code (NaN and infinity where the format has them).
    pub fn decode(self, code: u8) -> f32 {
        let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
        let (ebits, mbits, bias) = match self {
            Fp8::E4M3 => (4u32, 3u32, 7i32),
            Fp8::E5M2 => (5, 2, 15),
        };
        let e = u32::from(code >> mbits) & ((1 << ebits) - 1);
        let m = u32::from(code) & ((1 << mbits) - 1);
        let top = (1 << ebits) - 1;
        if e == top {
            match self {
                Fp8::E4M3 if m == (1 << mbits) - 1 => return f32::NAN,
                Fp8::E4M3 => {}
                Fp8::E5M2 => {
                    return if m == 0 {
                        sign * f32::INFINITY
                    } else {
                        f32::NAN
                    }
                }
            }
        }
        let frac = m as f32 / (1 << mbits) as f32;
        if e == 0 {
            sign * frac * 2f32.powi(1 - bias)
        } else {
            sign * (1.0 + frac) * 2f32.powi(e as i32 - bias)
        }
    }

    /// The code nearest to `x` (ties to the even code), saturating at the largest finite value;
    /// NaN encodes as the format's NaN.
    pub fn encode(self, x: f32) -> u8 {
        if x.is_nan() {
            return match self {
                Fp8::E4M3 => 0x7f,
                Fp8::E5M2 => 0x7e,
            };
        }
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
        let a = x.abs().min(self.max());
        // every finite non-negative code is below this limit, and the values rise with the code
        let limit: u8 = match self {
            Fp8::E4M3 => 0x7e,
            Fp8::E5M2 => 0x7b,
        };
        let (mut lo, mut hi) = (0u8, limit);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.decode(mid) < a {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // `lo` is the first code at or above `a`; the code below may be nearer
        let up = lo;
        let down = lo.saturating_sub(1);
        let (du, dd) = ((self.decode(up) - a).abs(), (a - self.decode(down)).abs());
        let pick = if du < dd || (du == dd && up % 2 == 0) {
            up
        } else {
            down
        };
        sign | pick
    }
}

/// Encode whole blocks of 32 values.
pub fn encode(fmt: Fp8, data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / QK * BLOCK);
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        // the scale the decoder will see is the f16 one, so divide by that
        let d = f32_to_f16(amax / fmt.max());
        let df = f16_to_f32(d);
        let id = if df != 0.0 { 1.0 / df } else { 0.0 };
        out.extend_from_slice(&d.to_le_bytes());
        out.extend(blk.iter().map(|&v| fmt.encode(v * id)));
    }
    out
}

/// Decode whole blocks to f32.
pub fn decode(fmt: Fp8, bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK * QK);
    for b in bytes.chunks_exact(BLOCK) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        out.extend(b[2..].iter().map(|&c| d * fmt.decode(c)));
    }
    out
}

#[cfg(test)]
mod tests;
