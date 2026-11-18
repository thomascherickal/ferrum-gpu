//! Q2_K (ggml type 10) encoder: 256-value super-blocks of sixteen 16-value sub-blocks, 2-bit
//! quants with a 4-bit scale and a 4-bit min per sub-block, and an f16 super scale and super min.
//! 84 bytes per block: `scales[16]`, `qs[64]`, `d`, `dmin` (decoded by `blocks::dequant`).
//!
//! The search for each sub-block's scale and min follows ggml's reference quantiser
//! (`quantize_row_q2_K_ref` with `make_qkx2_quants`, no importance matrix): the same candidate
//! scales, the same least-squares fit, the same rounding, in f32.

use crate::f16::{f16_to_f32, f32_to_f16};
use crate::QK_K;

/// Bytes of one super-block.
pub const BLOCK: usize = 16 + QK_K / 4 + 4;

/// ggml's `nearest_int`: round half to even through the float-magic trick.
fn nearest_int(x: f32) -> i32 {
    let bits = (x + 12_582_912.0).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

/// The best `(scale, min)` for 16 values at `nmax = 3` levels (ggml's `make_qkx2_quants` with
/// `rmin = -0.5`, `rdelta = 0.1`, `nstep = 15`, mean-absolute-deviation), writing the quants to `l`.
/// The returned min is already negated like ggml's `*the_min`.
fn fit(x: &[f32; 16], l: &mut [u8; 16]) -> (f32, f32) {
    const NMAX: i32 = 3;
    let weights: [f32; 16] = std::array::from_fn(|i| x[i].abs());
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..16 {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.fill(0);
        return (0.0, -min);
    }
    let mut iscale = NMAX as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..16 {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, NMAX);
        l[i] = q as u8;
        let diff = scale * q as f32 + min - x[i];
        best_mad += weights[i] * diff.abs();
    }
    for step in 0..=15 {
        iscale = (-0.5 + 0.1 * step as f32 + NMAX as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        let mut laux = [0u8; 16];
        for i in 0..16 {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, NMAX);
            laux[i] = q as u8;
            let (w, qf) = (weights[i], q as f32);
            sum_l += w * qf;
            sum_l2 += w * qf * qf;
            sum_xl += w * qf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..16 {
                let diff = this_scale * f32::from(laux[i]) + this_min - x[i];
                mad += weights[i] * diff.abs();
            }
            if mad < best_mad {
                *l = laux;
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

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

#[cfg(test)]
mod tests;
