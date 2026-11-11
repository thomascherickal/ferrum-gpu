// Provenance: the gradient formulas are the standard ones for RMSNorm, RoPE (the transpose of a
// rotation), grouped-query attention and SwiGLU, checked against finite differences here; the
// block order follows ferrum-cpu `llm_train/blocks.rs` (commit 56a9d3a). The code is our own.
//! Training forward/backward for [`Llama`] on the CPU: the oracle for the GPU trainer. Batched:
//! `ids` is `batch` sequences of length `t`. Tied heads are refused (train an untied copy).

use super::ops::{apply_rope, rmsnorm, rope_tables, silu, swiglu, RopeType};
use super::{Config, Llama};
use crate::grad::linear_bwd;
use crate::matmul::MatmulError;
use crate::nn;
use crate::tuned::matmul_tuned;

fn mm(a: &[f32], b: &[f32], (m, k, n): (usize, usize, usize), threads: usize) -> Vec<f32> {
    matmul_tuned(a, b, m, k, n, threads)
        .unwrap_or_else(|e: MatmulError| panic!("internal shape error: {e}"))
}

fn lin(x: &[f32], w: &[f32], b: &[f32], dims: (usize, usize, usize), threads: usize) -> Vec<f32> {
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

struct BlockCache {
    x: Vec<f32>,
    h1: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    probs: Vec<f32>,
    a: Vec<f32>,
    x2: Vec<f32>,
    h2: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
}

fn check(m: &Llama, ids: &[usize], targets: &[usize], batch: usize) -> Result<usize, String> {
    let c = &m.cfg;
    if m.tied {
        return Err("training a tied head is not supported: train an untied copy".into());
    }
    if batch == 0 || ids.is_empty() || !ids.len().is_multiple_of(batch) {
        return Err(format!(
            "{} tokens do not split into {batch} sequences",
            ids.len()
        ));
    }
    let t = ids.len() / batch;
    if t > c.ctx {
        return Err(format!("sequence length {t} exceeds the context {}", c.ctx));
    }
    if targets.len() != ids.len() {
        return Err(format!(
            "{} targets for {} tokens",
            targets.len(),
            ids.len()
        ));
    }
    if let Some(bad) = ids.iter().chain(targets).find(|&&i| i >= c.vocab) {
        return Err(format!("token {bad} is out of range for vocab {}", c.vocab));
    }
    Ok(t)
}

fn rope_dims(c: &Config, rows: usize, heads: usize) -> (usize, usize, usize, usize) {
    (rows, heads, c.head_dim, c.rope_dim)
}

/// Mean next-token cross-entropy over all positions and the gradient of every parameter
/// (returned as a [`Llama`] of the same shape; an absent bias has an empty gradient).
pub fn loss_and_grads(
    m: &Llama,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
    threads: usize,
) -> Result<(f32, Llama), String> {
    let t = check(m, ids, targets, batch)?;
    let (c, rows) = (&m.cfg, ids.len());
    let (d, hid, hd, qd, kvd) = (c.dim, c.hidden, c.head_dim, c.q_dim(), c.kv_dim());
    let shape = (batch, t, c.heads, c.kv_heads, hd);
    // RoPE tables for positions 0..t, repeated for every sequence of the batch.
    let (c1, s1) = rope_tables(0, t, c.rope_dim, c.rope_base, &m.rope);
    let (cos, sin) = (c1.repeat(batch), s1.repeat(batch));
    let tables = (&cos[..], &sin[..]);
    let mut x: Vec<f32> = ids
        .iter()
        .flat_map(|&i| m.tok[i * d..(i + 1) * d].iter().copied())
        .collect();
    let mut caches = Vec::with_capacity(m.blocks.len());
    for b in &m.blocks {
        let x_in = x.clone();
        let h1 = rmsnorm(&x, &b.attn_norm, rows, d, c.eps);
        let mut q = lin(&h1, &b.wq, &b.bq, (rows, d, qd), threads);
        let mut k = lin(&h1, &b.wk, &b.bk, (rows, d, kvd), threads);
        let v = lin(&h1, &b.wv, &b.bv, (rows, d, kvd), threads);
        apply_rope(&mut q, rope_dims(c, rows, c.heads), tables, c.rope_type);
        apply_rope(&mut k, rope_dims(c, rows, c.kv_heads), tables, c.rope_type);
        let (a, probs) = gqa_fwd(&q, &k, &v, shape);
        let x2 = nn::add(&x, &mm(&a, &b.wo, (rows, qd, d), threads));
        let h2 = rmsnorm(&x2, &b.ffn_norm, rows, d, c.eps);
        let gate = mm(&h2, &b.gate, (rows, d, hid), threads);
        let up = mm(&h2, &b.up, (rows, d, hid), threads);
        let act = swiglu(&gate, &up);
        x = nn::add(&x2, &mm(&act, &b.down, (rows, hid, d), threads));
        caches.push(BlockCache {
            x: x_in,
            h1,
            q,
            k,
            v,
            probs,
            a,
            x2,
            h2,
            gate,
            up,
            act,
        });
    }
    let hf = rmsnorm(&x, &m.out_norm, rows, d, c.eps);
    let logits = mm(&hf, &m.head, (rows, d, c.vocab), threads);
    let (loss, dlogits) = nn::cross_entropy(&logits, targets, c.vocab);

    let mut g = m.zeros_like();
    let (dhf, dhead, _) = linear_bwd(&hf, &m.head, &dlogits, (rows, d, c.vocab), threads);
    g.head = dhead;
    let (mut dx, dg) = rmsnorm_bwd(&x, &m.out_norm, &dhf, (rows, d), c.eps);
    g.out_norm = dg;
    for (bi, (b, ca)) in m.blocks.iter().zip(&caches).enumerate().rev() {
        let gb = &mut g.blocks[bi];
        // x_out = x2 + down(act)
        let (dact, ddown, _) = linear_bwd(&ca.act, &b.down, &dx, (rows, hid, d), threads);
        gb.down = ddown;
        let (dgate, dup) = swiglu_bwd(&ca.gate, &ca.up, &dact);
        let (dh2a, dwg, _) = linear_bwd(&ca.h2, &b.gate, &dgate, (rows, d, hid), threads);
        let (dh2b, dwu, _) = linear_bwd(&ca.h2, &b.up, &dup, (rows, d, hid), threads);
        gb.gate = dwg;
        gb.up = dwu;
        let (dx2_n, dgn) = rmsnorm_bwd(
            &ca.x2,
            &b.ffn_norm,
            &nn::add(&dh2a, &dh2b),
            (rows, d),
            c.eps,
        );
        gb.ffn_norm = dgn;
        let dx2 = nn::add(&dx, &dx2_n);
        // x2 = x + wo(a)
        let (da, dwo, _) = linear_bwd(&ca.a, &b.wo, &dx2, (rows, qd, d), threads);
        gb.wo = dwo;
        let (mut dq, mut dk, dv) = gqa_bwd((&ca.q, &ca.k, &ca.v), &ca.probs, &da, shape);
        rope_bwd(&mut dq, rope_dims(c, rows, c.heads), tables, c.rope_type);
        rope_bwd(&mut dk, rope_dims(c, rows, c.kv_heads), tables, c.rope_type);
        let (dh1a, dwq, dbq) = linear_bwd(&ca.h1, &b.wq, &dq, (rows, d, qd), threads);
        let (dh1b, dwk, dbk) = linear_bwd(&ca.h1, &b.wk, &dk, (rows, d, kvd), threads);
        let (dh1c, dwv, dbv) = linear_bwd(&ca.h1, &b.wv, &dv, (rows, d, kvd), threads);
        gb.wq = dwq;
        gb.wk = dwk;
        gb.wv = dwv;
        for (has, dst, db) in [
            (&b.bq, &mut gb.bq, dbq),
            (&b.bk, &mut gb.bk, dbk),
            (&b.bv, &mut gb.bv, dbv),
        ] {
            if !has.is_empty() {
                *dst = db;
            }
        }
        let dh1 = nn::add(&nn::add(&dh1a, &dh1b), &dh1c);
        let (dx_n, dg1) = rmsnorm_bwd(&ca.x, &b.attn_norm, &dh1, (rows, d), c.eps);
        gb.attn_norm = dg1;
        dx = nn::add(&dx2, &dx_n);
    }
    // Embedding gradient: rows summed per token id, in row order.
    for (r, &id) in ids.iter().enumerate() {
        for i in 0..d {
            g.tok[id * d + i] += dx[r * d + i];
        }
    }
    Ok((loss, g))
}
