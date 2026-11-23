use super::*;

/// Q5_0 (`(q - 16) * d`) and Q5_1 (`q * d + m`): `q` is the nibble plus bit `j` (first half) or
/// bit `j + 16` (second half) of the 32-bit `qh` as the fifth bit.
pub(crate) fn q5(b: &[u8], with_min: bool, out: &mut Vec<f32>) {
    let d = h(b, 0);
    let (m, at) = if with_min { (h(b, 2), 4) } else { (0.0, 2) };
    let qh = u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    let qs = &b[at + 4..at + 4 + QK / 2];
    let val = |x: u32| {
        if with_min {
            d * x as f32 + m
        } else {
            d * (x as i32 - 16) as f32
        }
    };
    out.extend(
        qs.iter()
            .enumerate()
            .map(|(j, &x)| val(u32::from(x & 0x0f) | (((qh >> j) & 1) << 4))),
    );
    out.extend(
        qs.iter()
            .enumerate()
            .map(|(j, &x)| val(u32::from(x >> 4) | (((qh >> (j + 16)) & 1) << 4))),
    );
}

/// The 6-bit scale and min of sub-block `j` (0..8) from the 12 packed bytes of a Q4_K/Q5_K block.
pub(super) fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

pub(super) fn q4_k(b: &[u8], out: &mut Vec<f32>) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let (scales, qs) = (&b[4..16], &b[16..16 + QK_K / 2]);
    for (pair, q) in qs.chunks_exact(32).enumerate() {
        let (s1, m1) = scale_min(2 * pair, scales);
        let (s2, m2) = scale_min(2 * pair + 1, scales);
        let (d1, n1, d2, n2) = (
            d * f32::from(s1),
            dmin * f32::from(m1),
            d * f32::from(s2),
            dmin * f32::from(m2),
        );
        out.extend(q.iter().map(|&x| d1 * f32::from(x & 0x0f) - n1));
        out.extend(q.iter().map(|&x| d2 * f32::from(x >> 4) - n2));
    }
}
