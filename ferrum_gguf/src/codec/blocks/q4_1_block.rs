use super::*;

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
        GGML_MXFP4 => (crate::mxfp4::BLOCK, QK),
        _ => return None,
    })
}

pub(crate) fn h(b: &[u8], at: usize) -> f32 {
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
            GGML_MXFP4 => out.extend(crate::mxfp4::decode(b)),
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
