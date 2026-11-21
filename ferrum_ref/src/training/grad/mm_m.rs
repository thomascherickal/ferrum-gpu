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

/// Transpose a row-major `[rows, cols]` matrix.
pub fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

pub(super) fn colsum(dy: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut s = vec![0.0; cols];
    for r in 0..rows {
        for c in 0..cols {
            s[c] += dy[r * cols + c];
        }
    }
    s
}

/// Gradient of `y = x w + b`: returns `(dx [m,k], dw [k,n], db [n])`.
pub fn linear_bwd(
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    (m, k, n): (usize, usize, usize),
    threads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dx = mm(dy, &transpose(w, k, n), (m, n, k), threads);
    let dw = mm(&transpose(x, m, k), dy, (k, m, n), threads);
    (dx, dw, colsum(dy, m, n))
}

/// LayerNorm backward: returns `(dx, dgain, dshift)`.
pub fn layernorm_bwd(
    x: &[f32],
    g: &[f32],
    dy: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut dx, mut dg, mut db) = (vec![0.0; rows * cols], vec![0.0; cols], vec![0.0; cols]);
    let n = cols as f32;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let is = 1.0 / (var + LN_EPS).sqrt();
        let xhat: Vec<f32> = row.iter().map(|&v| (v - mean) * is).collect();
        let dxhat: Vec<f32> = (0..cols).map(|c| dy[r * cols + c] * g[c]).collect();
        let (s1, s2) = (
            dxhat.iter().sum::<f32>() / n,
            dxhat.iter().zip(&xhat).map(|(a, b)| a * b).sum::<f32>() / n,
        );
        for c in 0..cols {
            dx[r * cols + c] = is * (dxhat[c] - s1 - xhat[c] * s2);
            dg[c] += dy[r * cols + c] * xhat[c];
            db[c] += dy[r * cols + c];
        }
    }
    (dx, dg, db)
}

/// Derivative of [`nn::gelu`] (exactly 0 or 1 beyond |x| = 10, where tanh has saturated).
pub fn gelu_grad(x: f32) -> f32 {
    if x.abs() >= 10.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    const K: f32 = 0.797_884_6;
    let t = (K * (x + 0.044715 * x * x * x)).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * K * (1.0 + 3.0 * 0.044715 * x * x)
}

/// Causal attention backward from the stored probabilities: returns `(dq, dk, dv)`.
pub fn attention_bwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    dout: &[f32],
    (batch, t, heads, hd): (usize, usize, usize, usize),
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let (mut dq, mut dk, mut dv) = (vec![0.0; q.len()], vec![0.0; k.len()], vec![0.0; v.len()]);
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..t {
                let p = &probs[((b * heads + h) * t + i) * t..][..t];
                let o = (b * t + i) * c + h * hd;
                let dp: Vec<f32> = (0..=i)
                    .map(|j| {
                        (0..hd)
                            .map(|d| dout[o + d] * v[(b * t + j) * c + h * hd + d])
                            .sum()
                    })
                    .collect();
                for (j, &pj) in p.iter().enumerate().take(i + 1) {
                    for d in 0..hd {
                        dv[(b * t + j) * c + h * hd + d] += pj * dout[o + d];
                    }
                }
                let dot: f32 = (0..=i).map(|j| p[j] * dp[j]).sum();
                for j in 0..=i {
                    let ds = p[j] * (dp[j] - dot) * scale;
                    for d in 0..hd {
                        dq[o + d] += ds * k[(b * t + j) * c + h * hd + d];
                        dk[(b * t + j) * c + h * hd + d] += ds * q[o + d];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

pub(super) struct BlockCache {
    pub(super) x: Vec<f32>,
    pub(super) h1: Vec<f32>,
    pub(super) q: Vec<f32>,
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) probs: Vec<f32>,
    pub(super) a: Vec<f32>,
    pub(super) x2: Vec<f32>,
    pub(super) h2: Vec<f32>,
    pub(super) f1: Vec<f32>,
    /// The FFN hidden activations as fed to `w2` (after dropout, when it is on).
    pub(super) act: Vec<f32>,
    /// The dropout multipliers applied to the activations, if dropout is on.
    pub(super) mask: Option<Vec<f32>>,
}

pub(super) fn lin(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    dims: (usize, usize, usize),
    threads: usize,
) -> Vec<f32> {
    nn::add_bias(&mm(x, w, dims, threads), b)
}
