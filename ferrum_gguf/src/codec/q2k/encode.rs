use super::*;

/// Encode whole super-blocks; the length must be a multiple of 256.
pub fn encode(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / QK_K * BLOCK);
    for blk in data.chunks_exact(QK_K) {
        let mut scales = [0.0f32; 16];
        let mut mins = [0.0f32; 16];
        let mut levels = [0u8; QK_K];
        for j in 0..16 {
            let sub: &[f32; 16] = blk[16 * j..16 * j + 16].try_into().expect("16 values");
            let mut l = [0u8; 16];
            (scales[j], mins[j]) = fit(sub, &mut l);
            levels[16 * j..16 * j + 16].copy_from_slice(&l);
        }
        let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = mins.iter().copied().fold(0.0f32, f32::max);
        let mut packed = [0u8; 16];
        let d = if max_scale > 0.0 {
            let iscale = 15.0 / max_scale;
            for j in 0..16 {
                packed[j] = nearest_int(iscale * scales[j]) as u8;
            }
            f32_to_f16(max_scale / 15.0)
        } else {
            0
        };
        let dmin = if max_min > 0.0 {
            let iscale = 15.0 / max_min;
            for j in 0..16 {
                packed[j] |= (nearest_int(iscale * mins[j]) as u8) << 4;
            }
            f32_to_f16(max_min / 15.0)
        } else {
            0
        };
        // re-quantise against the rounded scales the decoder will actually use
        let (d32, dmin32) = (f16_to_f32(d), f16_to_f32(dmin));
        for j in 0..16 {
            let sc = d32 * f32::from(packed[j] & 15);
            if sc == 0.0 {
                continue;
            }
            let dm = dmin32 * f32::from(packed[j] >> 4);
            for ii in 0..16 {
                let q = nearest_int((blk[16 * j + ii] + dm) / sc);
                levels[16 * j + ii] = q.clamp(0, 3) as u8;
            }
        }
        out.extend_from_slice(&packed);
        for half in (0..QK_K).step_by(128) {
            for l in 0..32 {
                out.push(
                    levels[half + l]
                        | (levels[half + l + 32] << 2)
                        | (levels[half + l + 64] << 4)
                        | (levels[half + l + 96] << 6),
                );
            }
        }
        out.extend_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&dmin.to_le_bytes());
    }
    out
}
