use super::*;

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
