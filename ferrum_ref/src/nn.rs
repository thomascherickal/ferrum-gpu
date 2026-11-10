// Provenance: semantics copied from ferrum-cpu (commit 56a9d3a): `activation.rs` (gelu),
// `train_transformer/kernels.rs` (ln_fwd, attn_fwd), `loss.rs` (softmax_cross_entropy,
// clamp_prob). Rewritten over plain slices; ours from here on.
//! Forward ops of the GPT-2 stack on row-major f32 slices: the parity oracle for the GPU kernels.

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

fn clamp_prob(p: f32) -> f32 {
    if p.is_nan() {
        p
    } else {
        p.max(1e-12)
    }
}

/// Mean softmax cross-entropy and its gradient `(p - onehot) / rows`. Targets must be `< vocab`.
pub fn cross_entropy(logits: &[f32], targets: &[usize], vocab: usize) -> (f32, Vec<f32>) {
    let rows = targets.len();
    let inv = 1.0 / rows as f32;
    let mut grad = vec![0.0; rows * vocab];
    let mut total = 0.0;
    for (i, &t) in targets.iter().enumerate() {
        let row = &logits[i * vocab..(i + 1) * vocab];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        total += -clamp_prob(probs[t]).ln();
        for j in 0..vocab {
            grad[i * vocab + j] = (probs[j] - if j == t { 1.0 } else { 0.0 }) * inv;
        }
    }
    (total * inv, grad)
}

/// Per row: `(log-probability of the target, whether the target is the first argmax)`.
pub fn score(logits: &[f32], targets: &[usize], vocab: usize) -> Vec<(f32, bool)> {
    targets
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = row.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
            let first_max = row.iter().position(|&x| x == max).unwrap_or(0);
            (row[t] - max - lse, first_max == t)
        })
        .collect()
}

/// `out[r] = tok[ids[r]] + pos[r % t]` for `ids.len()` rows of width `c`.
pub fn embed(tok: &[f32], pos: &[f32], ids: &[usize], t: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; ids.len() * c];
    for (r, &id) in ids.iter().enumerate() {
        for d in 0..c {
            out[r * c + d] = tok[id * c + d] + pos[(r % t) * c + d];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::rand_vec;

    #[test]
    fn gelu_known_values() {
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-5);
        assert!((gelu(-1.0) + 0.158_808).abs() < 1e-5);
        assert_eq!(gelu(20.0), 20.0);
    }

    #[test]
    fn bias_and_add() {
        assert_eq!(
            add_bias(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0]),
            vec![11.0, 22.0, 13.0, 24.0]
        );
        assert_eq!(add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
    }

    #[test]
    fn layernorm_normalises_each_row() {
        let x = rand_vec(3 * 8, 1);
        let y = layernorm(&x, &[1.0; 8], &[0.0; 8], 3, 8);
        for r in 0..3 {
            let row = &y[r * 8..(r + 1) * 8];
            let mean: f32 = row.iter().sum::<f32>() / 8.0;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
            assert!(mean.abs() < 1e-5 && (var - 1.0).abs() < 1e-3);
        }
        // gain and bias apply after normalising
        let y2 = layernorm(&x, &[2.0; 8], &[1.0; 8], 3, 8);
        assert!((y2[0] - (2.0 * y[0] + 1.0)).abs() < 1e-5);
    }

    #[test]
    fn attention_is_causal_and_rows_sum_to_one() {
        let (batch, t, heads, hd) = (2, 5, 2, 3);
        let n = batch * t * heads * hd;
        let (q, k, v) = (rand_vec(n, 1), rand_vec(n, 2), rand_vec(n, 3));
        let (out, probs) = attention(&q, &k, &v, (batch, t, heads, hd));
        assert_eq!(out.len(), n);
        for row in 0..batch * heads * t {
            let p = &probs[row * t..(row + 1) * t];
            let i = row % t;
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(
                p[i + 1..].iter().all(|&x| x == 0.0),
                "future positions must be masked"
            );
        }
        // the first token attends only to itself, so out == v there
        assert!((out[0] - v[0]).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_known_loss_and_gradient_sums_to_zero() {
        let (loss, g) = cross_entropy(&[0.0, 0.0, 0.0, 0.0], &[1], 4);
        assert!((loss - 4f32.ln()).abs() < 1e-6);
        assert!(g.iter().sum::<f32>().abs() < 1e-6);
        assert!((g[1] - (0.25 - 1.0)).abs() < 1e-6);
        let (_, g2) = cross_entropy(&rand_vec(2 * 6, 4), &[0, 5], 6);
        assert!(g2.iter().sum::<f32>().abs() < 1e-5);
    }

    #[test]
    fn score_logprob_and_argmax_flag() {
        let s = score(&[1.0, 3.0, 2.0, 5.0, 5.0, 1.0], &[1, 1], 3);
        assert!(s[0].1, "target 1 is the argmax of row 0");
        assert!(
            !s[1].1,
            "ties resolve to the first maximum, which is index 0"
        );
        let (loss, _) = cross_entropy(&[1.0, 3.0, 2.0], &[1], 3);
        assert!((s[0].0 + loss).abs() < 1e-5);
    }

    #[test]
    fn embed_adds_token_and_position_rows() {
        let tok = [1.0, 2.0, 10.0, 20.0]; // vocab 2, c 2
        let pos = [0.1, 0.2, 0.3, 0.4]; // t 2
        assert_eq!(
            embed(&tok, &pos, &[1, 0, 1], 2, 2),
            vec![10.1, 20.2, 1.3, 2.4, 10.1, 20.2]
        );
    }
}
