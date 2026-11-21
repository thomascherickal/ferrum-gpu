use super::*;

pub(super) fn mm(
    a: &[f32],
    b: &[f32],
    (m, k, n): (usize, usize, usize),
    threads: usize,
) -> Vec<f32> {
    matmul_tuned(a, b, m, k, n, threads)
        .unwrap_or_else(|e: MatmulError| panic!("internal shape error: {e}"))
}

pub(super) fn lin(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    dims: (usize, usize, usize),
    threads: usize,
) -> Vec<f32> {
    let y = mm(x, w, dims, threads);
    if b.is_empty() {
        y
    } else {
        nn::add_bias(&y, b)
    }
}

/// RMSNorm backward: returns `(dx, dgain)`. With `inv = (mean(x^2) + eps)^-1/2`,
/// `dx = inv * (dy*g) - x * inv^3 / n * sum(dy*g*x)` per row.
pub fn rmsnorm_bwd(
    x: &[f32],
    g: &[f32],
    dy: &[f32],
    (rows, cols): (usize, usize),
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (mut dx, mut dg) = (vec![0.0; rows * cols], vec![0.0; cols]);
    for r in 0..rows {
        let (xr, dyr) = (&x[r * cols..(r + 1) * cols], &dy[r * cols..(r + 1) * cols]);
        let ms = xr.iter().map(|&v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let dot: f32 = (0..cols).map(|c| dyr[c] * g[c] * xr[c]).sum();
        let k = inv * inv * inv * dot / cols as f32;
        for c in 0..cols {
            dx[r * cols + c] = inv * dyr[c] * g[c] - xr[c] * k;
            dg[c] += dyr[c] * xr[c] * inv;
        }
    }
    (dx, dg)
}

/// SwiGLU backward: returns `(dgate, dup)` for `act = silu(gate) * up`.
pub fn swiglu_bwd(gate: &[f32], up: &[f32], dact: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut dgate = Vec::with_capacity(gate.len());
    let mut dup = Vec::with_capacity(gate.len());
    for ((&g, &u), &d) in gate.iter().zip(up).zip(dact) {
        let s = 1.0 / (1.0 + (-g).exp());
        dgate.push(d * u * s * (1.0 + g * (1.0 - s)));
        dup.push(d * silu(g));
    }
    (dgate, dup)
}

/// Gradient through [`apply_rope`]: the transpose of a rotation is the rotation by the opposite
/// angle, so this applies the same tables with the sine negated.
pub fn rope_bwd(
    dy: &mut [f32],
    dims: (usize, usize, usize, usize),
    (cos, sin): (&[f32], &[f32]),
    ty: RopeType,
) {
    let neg: Vec<f32> = sin.iter().map(|s| -s).collect();
    apply_rope(dy, dims, (cos, &neg), ty);
}

/// Grouped-query causal attention over `batch` sequences of length `t`: returns
/// `(out [batch*t, heads*hd], probs [batch*heads*t, t])`.
pub fn gqa_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (batch, t, heads, kv, hd): (usize, usize, usize, usize, usize),
) -> (Vec<f32>, Vec<f32>) {
    let (qd, kvd, group) = (heads * hd, kv * hd, heads / kv);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; batch * t * qd];
    let mut probs = vec![0.0; batch * heads * t * t];
    for b in 0..batch {
        for h in 0..heads {
            let off = (h / group) * hd;
            for i in 0..t {
                let qi = &q[(b * t + i) * qd + h * hd..][..hd];
                let mut s: Vec<f32> = (0..=i)
                    .map(|j| {
                        let kj = &k[(b * t + j) * kvd + off..][..hd];
                        qi.iter().zip(kj).map(|(a, c)| a * c).sum::<f32>() * scale
                    })
                    .collect();
                let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                s.iter_mut().for_each(|x| {
                    *x = (*x - max).exp();
                    sum += *x;
                });
                let base = ((b * heads + h) * t + i) * t;
                for d in 0..hd {
                    let mut acc = 0.0;
                    for (j, sj) in s.iter().enumerate() {
                        let p = sj / sum;
                        probs[base + j] = p;
                        acc += p * v[(b * t + j) * kvd + off + d];
                    }
                    out[(b * t + i) * qd + h * hd + d] = acc;
                }
            }
        }
    }
    (out, probs)
}

/// Grouped-query attention backward from the stored probabilities: `(dq, dk, dv)`. Query heads
/// that share a KV head all accumulate into its `dk` and `dv`.
pub fn gqa_bwd(
    (q, k, v): (&[f32], &[f32], &[f32]),
    probs: &[f32],
    dout: &[f32],
    (batch, t, heads, kv, hd): (usize, usize, usize, usize, usize),
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (qd, kvd, group) = (heads * hd, kv * hd, heads / kv);
    let scale = 1.0 / (hd as f32).sqrt();
    let (mut dq, mut dk, mut dv) = (vec![0.0; q.len()], vec![0.0; k.len()], vec![0.0; v.len()]);
    for b in 0..batch {
        for h in 0..heads {
            let off = (h / group) * hd;
            for i in 0..t {
                let p = &probs[((b * heads + h) * t + i) * t..][..t];
                let o = (b * t + i) * qd + h * hd;
                let dp: Vec<f32> = (0..=i)
                    .map(|j| {
                        (0..hd)
                            .map(|d| dout[o + d] * v[(b * t + j) * kvd + off + d])
                            .sum()
                    })
                    .collect();
                for (j, &pj) in p.iter().enumerate().take(i + 1) {
                    for d in 0..hd {
                        dv[(b * t + j) * kvd + off + d] += pj * dout[o + d];
                    }
                }
                let dot: f32 = (0..=i).map(|j| p[j] * dp[j]).sum();
                for j in 0..=i {
                    let ds = p[j] * (dp[j] - dot) * scale;
                    for d in 0..hd {
                        dq[o + d] += ds * k[(b * t + j) * kvd + off + d];
                        dk[(b * t + j) * kvd + off + d] += ds * q[o + d];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}
