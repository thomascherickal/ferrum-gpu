use super::*;

/// Encode whole blocks of 32 values.
pub fn encode(fmt: Fp8, data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / QK * BLOCK);
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        // the scale the decoder will see is the f16 one, so divide by that
        let d = f32_to_f16(amax / fmt.max());
        let df = f16_to_f32(d);
        let id = if df != 0.0 { 1.0 / df } else { 0.0 };
        out.extend_from_slice(&d.to_le_bytes());
        out.extend(blk.iter().map(|&v| fmt.encode(v * id)));
    }
    out
}

/// Decode whole blocks to f32.
pub fn decode(fmt: Fp8, bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK * QK);
    for b in bytes.chunks_exact(BLOCK) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        out.extend(b[2..].iter().map(|&c| d * fmt.decode(c)));
    }
    out
}
