use super::*;

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
