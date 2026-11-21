use super::*;

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

pub(crate) struct BlockCache {
    pub(crate) x: Vec<f32>,
    pub(crate) h1: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) probs: Vec<f32>,
    pub(crate) a: Vec<f32>,
    pub(crate) x2: Vec<f32>,
    pub(crate) h2: Vec<f32>,
    pub(crate) f1: Vec<f32>,
    /// The FFN hidden activations as fed to `w2` (after dropout, when it is on).
    pub(crate) act: Vec<f32>,
    /// The dropout multipliers applied to the activations, if dropout is on.
    pub(crate) mask: Option<Vec<f32>>,
}

pub(crate) fn lin(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    dims: (usize, usize, usize),
    threads: usize,
) -> Vec<f32> {
    nn::add_bias(&mm(x, w, dims, threads), b)
}
