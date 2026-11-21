use super::*;

pub(super) fn err(m: impl Into<String>) -> GgufError {
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
