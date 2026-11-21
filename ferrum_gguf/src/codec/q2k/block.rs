use super::*;

/// Bytes of one super-block.
pub const BLOCK: usize = 16 + QK_K / 4 + 4;

/// ggml's `nearest_int`: round half to even through the float-magic trick.
pub(super) fn nearest_int(x: f32) -> i32 {
    let bits = (x + 12_582_912.0).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

/// The best `(scale, min)` for 16 values at `nmax = 3` levels (ggml's `make_qkx2_quants` with
/// `rmin = -0.5`, `rdelta = 0.1`, `nstep = 15`, mean-absolute-deviation), writing the quants to `l`.
/// The returned min is already negated like ggml's `*the_min`.
pub(super) fn fit(x: &[f32; 16], l: &mut [u8; 16]) -> (f32, f32) {
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
