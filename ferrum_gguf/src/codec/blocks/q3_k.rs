use super::*;

/// y = d * (scale - 32) * q with `q = low2 - (4 if the high bit is clear)`, same scan order as Q2_K;
/// the high-bit selector doubles after each shift group.
pub(crate) fn q3_k(b: &[u8], out: &mut Vec<f32>) {
    let (hmask, qs, sc) = (&b[..32], &b[32..96], &b[96..108]);
    let d = h(b, 108);
    let mut scales = [0i8; 16];
    for j in 0..8 {
        scales[j] = (sc[j] & 0x0f) as i8;
        scales[j + 8] = (sc[j] >> 4) as i8;
    }
    for (j, s) in scales.iter_mut().enumerate() {
        *s |= (((sc[8 + j / 4] >> (2 * (j % 4))) & 3) << 4) as i8;
    }
    let (mut is, mut m) = (0, 1u8);
    for half in 0..2 {
        let q = &qs[32 * half..32 * half + 32];
        for shift in [0u8, 2, 4, 6] {
            for run in 0..2 {
                let dl = d * f32::from(scales[is] - 32);
                is += 1;
                out.extend((16 * run..16 * run + 16).map(|l| {
                    let low = i32::from((q[l] >> shift) & 3);
                    dl * (low - if hmask[l] & m != 0 { 0 } else { 4 }) as f32
                }));
            }
            m <<= 1;
        }
    }
}
