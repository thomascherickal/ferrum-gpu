//! IEEE half-precision conversion (no dependencies).

/// Half-precision bits to f32 (handles subnormals, infinities and NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: normalise
            let shift = m.leading_zeros() - 21; // m has 10 significant bits at most
            let m = (m << shift) & 0x3ff;
            let e = 127 - 15 + 1 - shift;
            sign | (e << 23) | (m << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 to half-precision bits, round to nearest even; overflow gives infinity.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = (rem > mid || (rem == mid && (half & 1) == 1)) as u32;
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let round = (rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1)) as u32;
    sign | (half + round) as u16 // a carry into the exponent is correct (rounds up a binade)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1e9), 0x7c00); // overflow to infinity
        assert_eq!(f32_to_f16(1e-30), 0); // underflow to zero
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn every_half_round_trips_exactly() {
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if x.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(x), h, "half {h:#06x} = {x}");
        }
    }

    #[test]
    fn rounding_is_to_nearest_even() {
        // 1 + 2^-11 is exactly halfway between 1.0 and the next half; ties go to even (1.0).
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        // 1 + 3*2^-11 is halfway between 0x3c01 and 0x3c02; ties go to even (0x3c02).
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3c01);
    }
}
