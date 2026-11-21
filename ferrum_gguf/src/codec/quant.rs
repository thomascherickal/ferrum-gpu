//! Tensor element encodings: sizes, decode to f32 and encode from f32 (F32, F16, Q8_0, Q4_0, Q2_K,
//! MXFP4).

use crate::blocks::{dequant, layout};
use crate::f16::{f16_to_f32, f32_to_f16};
use crate::reader::GgufError;
use crate::{
    GGML_F16, GGML_F32, GGML_I16, GGML_I32, GGML_MXFP4, GGML_Q2_K, GGML_Q4_0, GGML_Q8_0, QK,
};

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
        GGML_F16 | GGML_I16 => n.checked_mul(2).ok_or_else(|| err("tensor size overflows")),
        GGML_I32 => n.checked_mul(4).ok_or_else(|| err("tensor size overflows")),
        GGML_Q8_0 => blocks(2 + QK),
        GGML_Q4_0 => blocks(2 + QK / 2),
        other if layout(other).is_some() => {
            let (bytes, elems) = layout(other).expect("checked");
            if !n.is_multiple_of(elems) {
                return Err(err(format!(
                    "{n} elements is not a multiple of the block size {elems}"
                )));
            }
            (n / elems)
                .checked_mul(bytes)
                .ok_or_else(|| err("tensor size overflows"))
        }
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
    if ty == GGML_I16 || ty == GGML_I32 {
        return Err(err(
            "integer tensors have no f32 form; read them with tensor_bytes",
        ));
    }
    if let Some(v) = dequant(ty, bytes) {
        return Ok(v);
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
    if layout(ty).is_some() && !matches!(ty, GGML_Q2_K | GGML_MXFP4) {
        return Err(err(format!(
            "tensor type {ty} is read-only: it can be loaded but not written"
        )));
    }
    let size = nbytes(ty, data.len())?; // also refuses a length that is not whole blocks
    match ty {
        GGML_Q2_K => return Ok(crate::q2k::encode(data)),
        GGML_MXFP4 => return Ok(crate::mxfp4::encode(data)),
        _ => {}
    }
    let mut out = Vec::with_capacity(size);
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
mod tests;
