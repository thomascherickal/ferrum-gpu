// Provenance: the block layouts and dequantization formulas follow ferrum-cpu
// ferrum_core/src/gguf/dequant.rs and format.rs at commit dbf6392c8883f293ac64b28a29c55e823179b257
// (2026-11-11), which follow ggml. Rewritten here over byte slices; our own code from here on.
//! Read-only block formats: Q4_1, Q5_0, Q5_1, Q8_1 and the k-quants Q2_K to Q6_K. These are
//! dequantized to f32 on load; we do not write them.

use crate::f16::f16_to_f32;
use crate::{
    GGML_Q2_K, GGML_Q3_K, GGML_Q4_1, GGML_Q4_K, GGML_Q5_0, GGML_Q5_1, GGML_Q5_K, GGML_Q6_K,
    GGML_Q8_1, QK, QK_K,
};

/// Bytes per block of Q4_1 (`d`, `m`, 16 nibble bytes).
pub const Q4_1_BLOCK: usize = 2 + 2 + QK / 2;
/// Bytes per block of Q5_0 (`d`, 4 high-bit bytes, 16 nibble bytes).
pub const Q5_0_BLOCK: usize = 2 + 4 + QK / 2;
/// Bytes per block of Q5_1 (`d`, `m`, 4 high-bit bytes, 16 nibble bytes).
pub const Q5_1_BLOCK: usize = 2 + 2 + 4 + QK / 2;
/// Bytes per block of Q8_1 (`d`, `s`, 32 bytes).
pub const Q8_1_BLOCK: usize = 2 + 2 + QK;
/// Bytes per super-block of Q2_K: scales, 2-bit quants, `d`, `dmin`.
pub const Q2_K_BLOCK: usize = 16 + QK_K / 4 + 2 + 2;
/// Bytes per super-block of Q3_K: high-bit mask, quants, packed scales, `d`.
pub const Q3_K_BLOCK: usize = QK_K / 8 + QK_K / 4 + 12 + 2;
/// Bytes per super-block of Q4_K: `d`, `dmin`, packed scales, 4-bit quants.
pub const Q4_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 2;
/// Bytes per super-block of Q5_K: Q4_K plus the high bits.
pub const Q5_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
/// Bytes per super-block of Q6_K: low 4 bits, high 2 bits, scales, `d`.
pub const Q6_K_BLOCK: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

/// `(bytes per block, elements per block)` of a read-only block type.
pub fn layout(ty: u32) -> Option<(usize, usize)> {
    Some(match ty {
        GGML_Q4_1 => (Q4_1_BLOCK, QK),
        GGML_Q5_0 => (Q5_0_BLOCK, QK),
        GGML_Q5_1 => (Q5_1_BLOCK, QK),
        GGML_Q8_1 => (Q8_1_BLOCK, QK),
        GGML_Q2_K => (Q2_K_BLOCK, QK_K),
        GGML_Q3_K => (Q3_K_BLOCK, QK_K),
        GGML_Q4_K => (Q4_K_BLOCK, QK_K),
        GGML_Q5_K => (Q5_K_BLOCK, QK_K),
        GGML_Q6_K => (Q6_K_BLOCK, QK_K),
        _ => return None,
    })
}

fn h(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// Dequantize `raw` (whole blocks of `ty`) to f32. `None` if `ty` is not one of these types.
pub fn dequant(ty: u32, raw: &[u8]) -> Option<Vec<f32>> {
    let (bytes, elems) = layout(ty)?;
    let mut out = Vec::with_capacity(raw.len() / bytes * elems);
    for b in raw.chunks_exact(bytes) {
        match ty {
            GGML_Q4_1 => q4_1(b, &mut out),
            GGML_Q5_0 => q5(b, false, &mut out),
            GGML_Q5_1 => q5(b, true, &mut out),
            GGML_Q8_1 => q8_1(b, &mut out),
            GGML_Q2_K => q2_k(b, &mut out),
            GGML_Q3_K => q3_k(b, &mut out),
            GGML_Q4_K => q4_k(b, &mut out),
            GGML_Q5_K => q5_k(b, &mut out),
            _ => q6_k(b, &mut out),
        }
    }
    Some(out)
}

fn q8_1(b: &[u8], out: &mut Vec<f32>) {
    let d = h(b, 0); // b[2..4] is the block sum, used only by dot products
    out.extend(b[4..4 + QK].iter().map(|&q| d * f32::from(q as i8)));
}

fn q4_1(b: &[u8], out: &mut Vec<f32>) {
    let (d, m) = (h(b, 0), h(b, 2));
    let qs = &b[4..4 + QK / 2];
    out.extend(qs.iter().map(|&x| d * f32::from(x & 0x0f) + m));
    out.extend(qs.iter().map(|&x| d * f32::from(x >> 4) + m));
}

/// Q5_0 (`(q - 16) * d`) and Q5_1 (`q * d + m`): `q` is the nibble plus bit `j` (first half) or
/// bit `j + 16` (second half) of the 32-bit `qh` as the fifth bit.
fn q5(b: &[u8], with_min: bool, out: &mut Vec<f32>) {
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
fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn q4_k(b: &[u8], out: &mut Vec<f32>) {
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

fn q5_k(b: &[u8], out: &mut Vec<f32>) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let (scales, qh, qs) = (
        &b[4..16],
        &b[16..16 + QK_K / 8],
        &b[16 + QK_K / 8..16 + QK_K / 8 + QK_K / 2],
    );
    for (pair, q) in qs.chunks_exact(32).enumerate() {
        let (s1, m1) = scale_min(2 * pair, scales);
        let (s2, m2) = scale_min(2 * pair + 1, scales);
        let (d1, n1, d2, n2) = (
            d * f32::from(s1),
            dmin * f32::from(m1),
            d * f32::from(s2),
            dmin * f32::from(m2),
        );
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

fn q6_k(b: &[u8], out: &mut Vec<f32>) {
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
fn q2_k(b: &[u8], out: &mut Vec<f32>) {
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

/// y = d * (scale - 32) * q with `q = low2 - (4 if the high bit is clear)`, same scan order as Q2_K;
/// the high-bit selector doubles after each shift group.
fn q3_k(b: &[u8], out: &mut Vec<f32>) {
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

#[cfg(test)]
mod tests;
