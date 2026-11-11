// Provenance: the gradient formulas (LayerNorm, attention, GELU derivative incl. its |x| >= 10
// saturation) follow ferrum-cpu `train_transformer/kernels.rs`, `backward.rs` and `activation.rs`
// (commit 56a9d3a); rewritten over plain slices, ours from here on.
//! Training forward/backward for [`Gpt2`] on the CPU: the oracle for the GPU trainer and the
//! CPU baseline for speed comparisons. Batched: `ids` is `batch` sequences of length `t`.

use crate::gpt2::{Block, Gpt2};
use crate::matmul::MatmulError;
use crate::nn::{self, LN_EPS};
use crate::tuned::matmul_tuned;

fn mm(a: &[f32], b: &[f32], (m, k, n): (usize, usize, usize), threads: usize) -> Vec<f32> {
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

fn colsum(dy: &[f32], rows: usize, cols: usize) -> Vec<f32> {
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
    f1: Vec<f32>,
    act: Vec<f32>,
}

fn lin(x: &[f32], w: &[f32], b: &[f32], dims: (usize, usize, usize), threads: usize) -> Vec<f32> {
    nn::add_bias(&mm(x, w, dims, threads), b)
}

fn check(m: &Gpt2, ids: &[usize], targets: &[usize], batch: usize) -> Result<usize, String> {
    let c = &m.cfg;
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

/// Mean next-token cross-entropy over all positions and the gradient of every parameter
/// (returned as a [`Gpt2`] of the same shape). `ids[i]` is predicted to be followed by
/// `targets[i]`. `threads` only splits matmuls, so the result is thread-count independent.
pub fn loss_and_grads(
    m: &Gpt2,
    ids: &[usize],
    targets: &[usize],
    batch: usize,
    threads: usize,
) -> Result<(f32, Gpt2), String> {
    let t = check(m, ids, targets, batch)?;
    let (c, rows) = (&m.cfg, ids.len());
    let (d, hid, hd) = (c.dim, c.hidden, c.head_dim());
    // Forward with caches.
    let mut x = nn::embed(&m.tok, &m.pos, ids, t, d);
    let mut caches = Vec::with_capacity(m.blocks.len());
    for b in &m.blocks {
        let x_in = x.clone();
        let h1 = nn::layernorm(&x, &b.ln1_g, &b.ln1_b, rows, d);
        let q = lin(&h1, &b.wq, &b.bq, (rows, d, d), threads);
        let k = lin(&h1, &b.wk, &b.bk, (rows, d, d), threads);
        let v = lin(&h1, &b.wv, &b.bv, (rows, d, d), threads);
        let (a, probs) = nn::attention(&q, &k, &v, (batch, t, c.heads, hd));
        let x2 = nn::add(&x, &lin(&a, &b.wo, &b.bo, (rows, d, d), threads));
        let h2 = nn::layernorm(&x2, &b.ln2_g, &b.ln2_b, rows, d);
        let f1 = lin(&h2, &b.w1, &b.b1, (rows, d, hid), threads);
        let act: Vec<f32> = f1.iter().map(|&v| nn::gelu(v)).collect();
        x = nn::add(&x2, &lin(&act, &b.w2, &b.b2, (rows, hid, d), threads));
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
            f1,
            act,
        });
    }
    let hf = nn::layernorm(&x, &m.lnf_g, &m.lnf_b, rows, d);
    let logits = mm(&hf, &m.head, (rows, d, c.vocab), threads);
    let (loss, dlogits) = nn::cross_entropy(&logits, targets, c.vocab);

    // Backward.
    let mut g = m.zeros_like();
    let (dhf, dhead, _) = linear_bwd(&hf, &m.head, &dlogits, (rows, d, c.vocab), threads);
    g.head = dhead;
    let (mut dx, dg, db) = layernorm_bwd(&x, &m.lnf_g, &dhf, rows, d);
    g.lnf_g = dg;
    g.lnf_b = db;
    for (bi, (b, ca)) in m.blocks.iter().zip(&caches).enumerate().rev() {
        let gb: &mut Block = &mut g.blocks[bi];
        // x_out = x2 + f2
        let (dact, dw2, db2) = linear_bwd(&ca.act, &b.w2, &dx, (rows, hid, d), threads);
        gb.w2 = dw2;
        gb.b2 = db2;
        let df1: Vec<f32> = dact
            .iter()
            .zip(&ca.f1)
            .map(|(&da, &f)| da * gelu_grad(f))
            .collect();
        let (dh2, dw1, db1) = linear_bwd(&ca.h2, &b.w1, &df1, (rows, d, hid), threads);
        gb.w1 = dw1;
        gb.b1 = db1;
        let (dx2_ln, dg2, dbl2) = layernorm_bwd(&ca.x2, &b.ln2_g, &dh2, rows, d);
        gb.ln2_g = dg2;
        gb.ln2_b = dbl2;
        let dx2 = nn::add(&dx, &dx2_ln);
        // x2 = x + o
        let (da, dwo, dbo) = linear_bwd(&ca.a, &b.wo, &dx2, (rows, d, d), threads);
        gb.wo = dwo;
        gb.bo = dbo;
        let (dq, dk, dv) =
            attention_bwd(&ca.q, &ca.k, &ca.v, &ca.probs, &da, (batch, t, c.heads, hd));
        let (dh1a, dwq, dbq) = linear_bwd(&ca.h1, &b.wq, &dq, (rows, d, d), threads);
        let (dh1b, dwk, dbk) = linear_bwd(&ca.h1, &b.wk, &dk, (rows, d, d), threads);
        let (dh1c, dwv, dbv) = linear_bwd(&ca.h1, &b.wv, &dv, (rows, d, d), threads);
        gb.wq = dwq;
        gb.bq = dbq;
        gb.wk = dwk;
        gb.bk = dbk;
        gb.wv = dwv;
        gb.bv = dbv;
        let dh1 = nn::add(&nn::add(&dh1a, &dh1b), &dh1c);
        let (dx_ln, dg1, dbl1) = layernorm_bwd(&ca.x, &b.ln1_g, &dh1, rows, d);
        gb.ln1_g = dg1;
        gb.ln1_b = dbl1;
        dx = nn::add(&dx2, &dx_ln);
    }
    // Embedding gradients: rows are summed per token id and per position, in row order.
    for (r, &id) in ids.iter().enumerate() {
        for i in 0..d {
            g.tok[id * d + i] += dx[r * d + i];
            g.pos[(r % t) * d + i] += dx[r * d + i];
        }
    }
    Ok((loss, g))
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index loops read clearer in element-wise finite-difference checks
mod tests {
    use super::*;
    use crate::gpt2::Config;
    use crate::matmul::rand_vec;

    fn tiny() -> Config {
        Config {
            vocab: 7,
            ctx: 5,
            dim: 4,
            heads: 2,
            layers: 2,
            hidden: 6,
        }
    }

    fn data(cfg: &Config, batch: usize, t: usize) -> (Vec<usize>, Vec<usize>) {
        let n = batch * t;
        (
            (0..n).map(|i| (i * 3 + 1) % cfg.vocab).collect(),
            (0..n).map(|i| (i * 5 + 2) % cfg.vocab).collect(),
        )
    }

    #[test]
    fn linear_bwd_matches_finite_differences() {
        let (m, k, n) = (3, 4, 5);
        let (x, w, dy) = (rand_vec(m * k, 1), rand_vec(k * n, 2), rand_vec(m * n, 3));
        let (dx, dw, db) = linear_bwd(&x, &w, &dy, (m, k, n), 1);
        let b = rand_vec(n, 4);
        let loss = |x: &[f32], w: &[f32], b: &[f32]| -> f32 {
            nn::add_bias(&mm(x, w, (m, k, n), 1), b)
                .iter()
                .zip(&dy)
                .map(|(a, d)| a * d)
                .sum()
        };
        let h = 1e-2;
        for i in 0..x.len() {
            let (mut p, mut q) = (x.clone(), x.clone());
            p[i] += h;
            q[i] -= h;
            assert!(
                (dx[i] - (loss(&p, &w, &b) - loss(&q, &w, &b)) / (2.0 * h)).abs() < 1e-3,
                "dx[{i}]"
            );
        }
        for i in 0..w.len() {
            let (mut p, mut q) = (w.clone(), w.clone());
            p[i] += h;
            q[i] -= h;
            assert!(
                (dw[i] - (loss(&x, &p, &b) - loss(&x, &q, &b)) / (2.0 * h)).abs() < 1e-3,
                "dw[{i}]"
            );
        }
        for i in 0..n {
            let (mut p, mut q) = (b.clone(), b.clone());
            p[i] += h;
            q[i] -= h;
            assert!(
                (db[i] - (loss(&x, &w, &p) - loss(&x, &w, &q)) / (2.0 * h)).abs() < 1e-3,
                "db[{i}]"
            );
        }
    }

    #[test]
    fn layernorm_bwd_matches_finite_differences() {
        let (rows, cols) = (3, 6);
        let (x, g, b, dy) = (
            rand_vec(rows * cols, 5),
            rand_vec(cols, 6),
            rand_vec(cols, 7),
            rand_vec(rows * cols, 8),
        );
        let (dx, dg, db) = layernorm_bwd(&x, &g, &dy, rows, cols);
        let loss = |x: &[f32], g: &[f32], b: &[f32]| -> f32 {
            nn::layernorm(x, g, b, rows, cols)
                .iter()
                .zip(&dy)
                .map(|(a, d)| a * d)
                .sum()
        };
        let h = 1e-2;
        let fd = |f: &dyn Fn(&[f32]) -> f32, v: &[f32], i: usize| {
            let (mut p, mut q) = (v.to_vec(), v.to_vec());
            p[i] += h;
            q[i] -= h;
            (f(&p) - f(&q)) / (2.0 * h)
        };
        for i in 0..x.len() {
            assert!(
                (dx[i] - fd(&|v| loss(v, &g, &b), &x, i)).abs() < 2e-3,
                "dx[{i}]"
            );
        }
        for i in 0..cols {
            assert!(
                (dg[i] - fd(&|v| loss(&x, v, &b), &g, i)).abs() < 2e-3,
                "dg[{i}]"
            );
            assert!(
                (db[i] - fd(&|v| loss(&x, &g, v), &b, i)).abs() < 2e-3,
                "db[{i}]"
            );
        }
    }

    #[test]
    fn gelu_grad_matches_finite_differences_and_saturates() {
        for x in [-9.0f32, -3.0, -1.0, -0.2, 0.0, 0.5, 2.0, 7.0] {
            let h = 1e-3;
            let num = (nn::gelu(x + h) - nn::gelu(x - h)) / (2.0 * h);
            assert!((gelu_grad(x) - num).abs() < 2e-3, "gelu'({x})");
        }
        assert_eq!(gelu_grad(10.0), 1.0);
        assert_eq!(gelu_grad(-10.0), 0.0);
        assert_eq!(gelu_grad(1e9), 1.0);
        assert_eq!(gelu_grad(0.0), 0.5);
    }

    #[test]
    fn attention_bwd_matches_finite_differences() {
        let shape = (2usize, 4usize, 2usize, 3usize);
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        let (q, k, v, dout) = (
            rand_vec(n, 11),
            rand_vec(n, 12),
            rand_vec(n, 13),
            rand_vec(n, 14),
        );
        let (_, probs) = nn::attention(&q, &k, &v, shape);
        let (dq, dk, dv) = attention_bwd(&q, &k, &v, &probs, &dout, shape);
        let loss = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
            nn::attention(q, k, v, shape)
                .0
                .iter()
                .zip(&dout)
                .map(|(a, d)| a * d)
                .sum()
        };
        let h = 1e-2;
        for i in 0..n {
            let step = |v: &[f32]| {
                let (mut p, mut q2) = (v.to_vec(), v.to_vec());
                p[i] += h;
                q2[i] -= h;
                (p, q2)
            };
            let (p, m) = step(&q);
            assert!(
                (dq[i] - (loss(&p, &k, &v) - loss(&m, &k, &v)) / (2.0 * h)).abs() < 2e-3,
                "dq[{i}]"
            );
            let (p, m) = step(&k);
            assert!(
                (dk[i] - (loss(&q, &p, &v) - loss(&q, &m, &v)) / (2.0 * h)).abs() < 2e-3,
                "dk[{i}]"
            );
            let (p, m) = step(&v);
            assert!(
                (dv[i] - (loss(&q, &k, &p) - loss(&q, &k, &m)) / (2.0 * h)).abs() < 2e-3,
                "dv[{i}]"
            );
        }
    }

    #[test]
    fn whole_model_gradients_match_finite_differences_for_every_parameter_group() {
        let cfg = tiny();
        let m = Gpt2::random(cfg, 21);
        let (batch, t) = (2, 4);
        let (ids, targets) = data(&cfg, batch, t);
        let (loss, g) = loss_and_grads(&m, &ids, &targets, batch, 1).unwrap();
        assert!(loss.is_finite() && loss > 0.0);
        let h = 1e-2;
        let eval = |m: &Gpt2| loss_and_grads(m, &ids, &targets, batch, 1).unwrap().0;
        let n_params = m.params().len();
        let mut checked = 0;
        for pi in 0..n_params {
            let len = m.params()[pi].len();
            // Check up to 6 spread-out elements of every tensor (all of them when small).
            let picks: Vec<usize> = if len <= 6 {
                (0..len).collect()
            } else {
                (0..6).map(|j| j * (len - 1) / 5).collect()
            };
            for &i in &picks {
                let (mut p, mut q) = (m.clone(), m.clone());
                p.params_mut()[pi][i] += h;
                q.params_mut()[pi][i] -= h;
                let num = (eval(&p) - eval(&q)) / (2.0 * h);
                let ana = g.params()[pi][i];
                assert!(
                    (ana - num).abs() < 4e-3 + 2e-2 * num.abs(),
                    "param {pi}[{i}]: analytic {ana} vs numeric {num}"
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} elements checked");
        let norm: f32 = g
            .params()
            .iter()
            .flat_map(|p| p.iter())
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(norm > 1e-3, "gradient is suspiciously zero");
    }

    #[test]
    fn token_embedding_gradient_sums_repeated_tokens() {
        let cfg = tiny();
        let m = Gpt2::random(cfg, 22);
        // The same token three times: its row is the sum of three position contributions.
        let ids = vec![2usize, 2, 2, 5];
        let targets = vec![1usize, 3, 4, 0];
        let (_, g) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
        assert!(g.tok[2 * 4..3 * 4].iter().any(|v| v.abs() > 1e-6));
        assert!(
            g.tok[..4].iter().all(|&v| v == 0.0),
            "unused token has no gradient"
        );
    }

    #[test]
    fn batching_equals_two_separate_sequences() {
        let cfg = tiny();
        let m = Gpt2::random(cfg, 23);
        let (ids, targets) = data(&cfg, 2, 3);
        let (l, g) = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
        let (l1, g1) = loss_and_grads(&m, &ids[..3], &targets[..3], 1, 1).unwrap();
        let (l2, g2) = loss_and_grads(&m, &ids[3..], &targets[3..], 1, 1).unwrap();
        assert!(
            (l - (l1 + l2) / 2.0).abs() < 1e-5,
            "mean loss is the average of the sequences"
        );
        for ((a, b), c) in g.params().iter().zip(g1.params()).zip(g2.params()) {
            for i in 0..a.len() {
                assert!(
                    (a[i] - (b[i] + c[i]) / 2.0).abs() < 1e-5,
                    "gradient is the average too"
                );
            }
        }
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let cfg = tiny();
        let m = Gpt2::random(cfg, 24);
        let (ids, targets) = data(&cfg, 2, 4);
        let one = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
        for t in [2, 5] {
            let other = loss_and_grads(&m, &ids, &targets, 2, t).unwrap();
            assert_eq!(one.0, other.0);
            assert_eq!(one.1, other.1, "bit-identical at {t} threads");
        }
    }

    #[test]
    fn input_validation() {
        let m = Gpt2::random(tiny(), 25);
        let e =
            |ids: &[usize], tg: &[usize], b: usize| loss_and_grads(&m, ids, tg, b, 1).unwrap_err();
        assert!(e(&[], &[], 1).contains("split"));
        assert!(e(&[1, 2, 3], &[1, 2, 3], 2).contains("split"));
        assert!(e(&[1; 6], &[1; 6], 1).contains("exceeds the context"));
        assert!(e(&[1, 2], &[1], 1).contains("targets"));
        assert!(e(&[7], &[1], 1).contains("out of range"));
        assert!(e(&[1], &[9], 1).contains("out of range"));
    }

    #[test]
    fn loss_matches_the_forward_oracle() {
        let cfg = tiny();
        let m = Gpt2::random(cfg, 26);
        let (ids, targets) = data(&cfg, 1, 5);
        let logits = m.forward_logits(&ids).unwrap();
        let (want, _) = nn::cross_entropy(&logits, &targets, cfg.vocab);
        let (got, _) = loss_and_grads(&m, &ids, &targets, 1, 1).unwrap();
        assert!((got - want).abs() < 1e-5);
    }
}
