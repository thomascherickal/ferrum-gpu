use super::*;

pub(crate) fn mm(
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

pub(crate) fn colsum(dy: &[f32], rows: usize, cols: usize) -> Vec<f32> {
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
