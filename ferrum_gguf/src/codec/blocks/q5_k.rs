use super::*;

pub(crate) fn q5_k(b: &[u8], out: &mut Vec<f32>) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let (scales, qh, qs) = (
        &b[4..16],
        &b[16..16 + QK_K / 8],
        &b[16 + QK_K / 8..16 + QK_K / 8 + QK_K / 2],
    );
    for (pair, q) in qs.chunks_exact(32).enumerate() {
        let (d1, n1, d2, n2) = pair_scales(pair, scales, d, dmin);
        let (u1, u2) = (1u8 << (2 * pair), 2u8 << (2 * pair));
        out.extend(
            q.iter().zip(qh).map(|(&x, &hb)| {
                d1 * f32::from((x & 0x0f) + if hb & u1 != 0 { 16 } else { 0 }) - n1
            }),
        );
        out.extend(
            q.iter()
                .zip(qh)
                .map(|(&x, &hb)| d2 * f32::from((x >> 4) + if hb & u2 != 0 { 16 } else { 0 }) - n2),
        );
    }
}

pub(super) fn q6_k(b: &[u8], out: &mut Vec<f32>) {
    let (ql, qh, scales) = (
        &b[..QK_K / 2],
        &b[QK_K / 2..QK_K / 2 + QK_K / 4],
        &b[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16],
    );
    let d = h(b, QK_K / 2 + QK_K / 4 + QK_K / 16);
    let start = out.len();
    out.resize(start + QK_K, 0.0);
    for half in 0..2 {
        let (qlo, qho, sco, oo) = (half * 64, half * 32, half * 8, start + half * 128);
        for l in 0..32 {
            let is = l / 16;
            let q = |lo: u8, hi: u8| ((i32::from(lo)) | (i32::from(hi & 3) << 4)) - 32;
            let q1 = q(ql[qlo + l] & 0x0f, qh[qho + l]);
            let q2 = q(ql[qlo + l + 32] & 0x0f, qh[qho + l] >> 2);
            let q3 = q(ql[qlo + l] >> 4, qh[qho + l] >> 4);
            let q4 = q(ql[qlo + l + 32] >> 4, qh[qho + l] >> 6);
            let sc = |k: usize| f32::from(scales[sco + k] as i8);
            out[oo + l] = d * sc(is) * q1 as f32;
            out[oo + l + 32] = d * sc(is + 2) * q2 as f32;
            out[oo + l + 64] = d * sc(is + 4) * q3 as f32;
            out[oo + l + 96] = d * sc(is + 6) * q4 as f32;
        }
    }
}

/// y = d * sc * q - dmin * m over ggml's scan order: two 128-element halves, shifts 0, 2, 4, 6,
/// two 16-element runs per shift.
pub(super) fn q2_k(b: &[u8], out: &mut Vec<f32>) {
    let (scales, qs) = (&b[..16], &b[16..16 + QK_K / 4]);
    let (d, dmin) = (h(b, 80), h(b, 82));
    let mut is = 0;
    for half in 0..2 {
        let q = &qs[32 * half..32 * half + 32];
        for shift in [0u8, 2, 4, 6] {
            for run in 0..2 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * f32::from(sc & 0x0f), dmin * f32::from(sc >> 4));
                out.extend(
                    q[16 * run..16 * run + 16]
                        .iter()
                        .map(|&x| dl * f32::from((x >> shift) & 3) - ml),
                );
            }
        }
    }
}
