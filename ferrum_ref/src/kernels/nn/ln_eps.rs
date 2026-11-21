/// LayerNorm epsilon (must match ferrum-cpu's `LN_EPS`).
pub const LN_EPS: f32 = 1e-5;
/// Causal mask value (not `-inf`; softmax subtracts the row max first).
pub const MASK: f32 = -1e9;

/// GELU, tanh approximation (identical to llama.cpp's `ggml_gelu`).
pub fn gelu(x: f32) -> f32 {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

/// `y[r,c] = x[r,c] + bias[c]` for `x` of `rows x bias.len()`.
pub fn add_bias(x: &[f32], bias: &[f32]) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| v + bias[i % bias.len()])
        .collect()
}

/// Element-wise sum.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Row-wise LayerNorm of `x` (`rows x cols`) with gain `g` and bias `b` (each `cols`).
pub fn layernorm(x: &[f32], g: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
        let is = 1.0 / (var + LN_EPS).sqrt();
        for c in 0..cols {
            y[r * cols + c] = (row[c] - mean) * is * g[c] + b[c];
        }
    }
    y
}

/// Causal multi-head attention forward. `q`,`k`,`v` are `[batch*t, heads*hd]`. Returns
/// `(out [batch*t, C], probs [batch*heads*t, t])`.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (batch, t, heads, hd): (usize, usize, usize, usize),
) -> (Vec<f32>, Vec<f32>) {
    let c = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; batch * t * c];
    let mut probs = vec![0.0; batch * heads * t * t];
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..t {
                let mut s = vec![MASK; t];
                for (j, sj) in s.iter_mut().enumerate().take(i + 1) {
                    let dot: f32 = (0..hd)
                        .map(|d| q[(b * t + i) * c + h * hd + d] * k[(b * t + j) * c + h * hd + d])
                        .sum();
                    *sj = dot * scale;
                }
                let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for sj in s.iter_mut() {
                    *sj = (*sj - max).exp();
                    sum += *sj;
                }
                let base = ((b * heads + h) * t + i) * t;
                for (j, sj) in s.iter().enumerate() {
                    probs[base + j] = sj / sum;
                }
                for d in 0..hd {
                    out[(b * t + i) * c + h * hd + d] = (0..t)
                        .map(|j| probs[base + j] * v[(b * t + j) * c + h * hd + d])
                        .sum();
                }
            }
        }
    }
    (out, probs)
}
