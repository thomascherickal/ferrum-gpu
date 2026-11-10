//! Forward kernels of the GPT-2 stack. Every function records one or more dispatches on
//! resident buffers and does not wait; read results with [`Gpu::read`].
//! ponytail: 1-D kernels cap at 65,535 workgroups x 64 invocations (4.1M elements) per call;
//! add a 2-D split when a tensor needs more.

use crate::device::{Gpu, GpuError};
use crate::runtime::groups_for;
use crate::tensor::GpuBuf;

fn err(msg: String) -> GpuError {
    GpuError(msg)
}

fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
    if b.len() == want {
        Ok(())
    } else {
        Err(err(format!(
            "{name}: buffer holds {} elements, expected {want}",
            b.len()
        )))
    }
}

fn one_d(gpu: &Gpu, n: usize) -> Result<(u32, u32), GpuError> {
    groups_for((n.max(1), 1), (64, 1), gpu.max_workgroups())
}

impl Gpu {
    /// Upload integer indices that must all be `< bound` (a GPU gather would silently clamp).
    pub fn upload_indices(&self, ids: &[usize], bound: usize) -> Result<GpuBuf, GpuError> {
        if let Some(bad) = ids.iter().find(|&&i| i >= bound) {
            return Err(err(format!(
                "index {bad} is out of range for {bound} entries"
            )));
        }
        let words: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.upload_raw(bytemuck::cast_slice(&words), words.len())
    }
}

/// `c = a + b`, element-wise.
pub fn add(gpu: &Gpu, a: &GpuBuf, b: &GpuBuf, c: &GpuBuf) -> Result<(), GpuError> {
    let n = a.len();
    need("add b", b, n)?;
    need("add c", c, n)?;
    let p = gpu.pipeline("add", include_str!("shaders/add.wgsl"))?;
    gpu.dispatch(&p, &[a, b, c], &[n as u32], one_d(gpu, n)?)
}

/// `y[i] = x[i] + bias[i % bias.len()]`.
pub fn add_bias(gpu: &Gpu, x: &GpuBuf, bias: &GpuBuf, y: &GpuBuf) -> Result<(), GpuError> {
    need("add_bias y", y, x.len())?;
    if bias.is_empty() || !x.len().is_multiple_of(bias.len()) {
        return Err(err(format!(
            "add_bias: {} elements is not a multiple of bias length {}",
            x.len(),
            bias.len()
        )));
    }
    let p = gpu.pipeline("bias_add", include_str!("shaders/bias_add.wgsl"))?;
    gpu.dispatch(
        &p,
        &[x, bias, y],
        &[x.len() as u32, bias.len() as u32],
        one_d(gpu, x.len())?,
    )
}

/// `y = gelu(x)` (tanh approximation).
pub fn gelu(gpu: &Gpu, x: &GpuBuf, y: &GpuBuf) -> Result<(), GpuError> {
    need("gelu y", y, x.len())?;
    let p = gpu.pipeline("gelu", include_str!("shaders/gelu.wgsl"))?;
    gpu.dispatch(&p, &[x, y], &[x.len() as u32], one_d(gpu, x.len())?)
}

/// Row-wise LayerNorm of `x` (`rows x cols`) into `y`.
pub fn layernorm(
    gpu: &Gpu,
    x: &GpuBuf,
    gain: &GpuBuf,
    shift: &GpuBuf,
    y: &GpuBuf,
    (rows, cols): (usize, usize),
) -> Result<(), GpuError> {
    need("layernorm x", x, rows * cols)?;
    need("layernorm y", y, rows * cols)?;
    need("layernorm gain", gain, cols)?;
    need("layernorm shift", shift, cols)?;
    let p = gpu.pipeline("layernorm", include_str!("shaders/layernorm.wgsl"))?;
    gpu.dispatch(
        &p,
        &[x, gain, shift, y],
        &[rows as u32, cols as u32],
        one_d(gpu, rows)?,
    )
}

/// Causal multi-head attention forward: fills `probs` (`batch*heads*t x t`) and `out` (`batch*t x heads*hd`).
pub fn attention(
    gpu: &Gpu,
    (q, k, v): (&GpuBuf, &GpuBuf, &GpuBuf),
    (probs, out): (&GpuBuf, &GpuBuf),
    (batch, t, heads, hd): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let c = heads * hd;
    for (name, b) in [("q", q), ("k", k), ("v", v), ("out", out)] {
        need(&format!("attention {name}"), b, batch * t * c)?;
    }
    need("attention probs", probs, batch * heads * t * t)?;
    let u = [batch as u32, t as u32, heads as u32, hd as u32];
    let p1 = gpu.pipeline("attn_probs", include_str!("shaders/attn_probs.wgsl"))?;
    gpu.dispatch(&p1, &[q, k, probs], &u, one_d(gpu, batch * heads * t)?)?;
    let p2 = gpu.pipeline("attn_pv", include_str!("shaders/attn_pv.wgsl"))?;
    gpu.dispatch(&p2, &[probs, v, out], &u, one_d(gpu, batch * t * c)?)
}

/// Softmax cross-entropy: fills `grad` (`rows x vocab`) and the one-element `loss` (mean).
/// `row_loss` is scratch of `rows` elements; `targets` come from [`Gpu::upload_indices`].
pub fn cross_entropy(
    gpu: &Gpu,
    logits: &GpuBuf,
    targets: &GpuBuf,
    (grad, row_loss, loss): (&GpuBuf, &GpuBuf, &GpuBuf),
    (rows, vocab): (usize, usize),
) -> Result<(), GpuError> {
    need("cross_entropy logits", logits, rows * vocab)?;
    need("cross_entropy targets", targets, rows)?;
    need("cross_entropy grad", grad, rows * vocab)?;
    need("cross_entropy row_loss", row_loss, rows)?;
    need("cross_entropy loss", loss, 1)?;
    if rows == 0 {
        return Err(err("cross_entropy: zero rows".into()));
    }
    let p1 = gpu.pipeline("ce_rows", include_str!("shaders/ce_rows.wgsl"))?;
    gpu.dispatch(
        &p1,
        &[logits, targets, grad, row_loss],
        &[rows as u32, vocab as u32],
        one_d(gpu, rows)?,
    )?;
    let p2 = gpu.pipeline("mean_loss", include_str!("shaders/mean_loss.wgsl"))?;
    gpu.dispatch(&p2, &[row_loss, loss], &[rows as u32], (1, 1))
}

/// Per-row `(log-probability of target, 1.0 if target is the first argmax else 0.0)` into `out` (`2*rows`).
pub fn score(
    gpu: &Gpu,
    logits: &GpuBuf,
    targets: &GpuBuf,
    out: &GpuBuf,
    (rows, vocab): (usize, usize),
) -> Result<(), GpuError> {
    need("score logits", logits, rows * vocab)?;
    need("score targets", targets, rows)?;
    need("score out", out, 2 * rows)?;
    let p = gpu.pipeline("score", include_str!("shaders/score.wgsl"))?;
    gpu.dispatch(
        &p,
        &[logits, targets, out],
        &[rows as u32, vocab as u32],
        one_d(gpu, rows)?,
    )
}

/// `out[r] = tok[ids[r]] + pos[r % t]` for `rows` rows of width `c`.
pub fn embed(
    gpu: &Gpu,
    tables: (&GpuBuf, &GpuBuf, &GpuBuf),
    out: &GpuBuf,
    dims: (usize, usize, usize),
) -> Result<(), GpuError> {
    embed_at(gpu, tables, out, dims, 0)
}

/// `out[r] = tok[ids[r]] + pos[(r + pos_offset) % t]`: row `r` sits at position `pos_offset + r`
/// (decode feeds one token at its absolute position).
pub fn embed_at(
    gpu: &Gpu,
    (tok, pos, ids): (&GpuBuf, &GpuBuf, &GpuBuf),
    out: &GpuBuf,
    (rows, c, t): (usize, usize, usize),
    pos_offset: usize,
) -> Result<(), GpuError> {
    need("embed ids", ids, rows)?;
    need("embed out", out, rows * c)?;
    if t == 0 || !pos.len().is_multiple_of(c.max(1)) || pos.len() / c.max(1) < t {
        return Err(err(format!(
            "embed: positions table of {} elements cannot cover t={t}",
            pos.len()
        )));
    }
    let p = gpu.pipeline("embed", include_str!("shaders/embed.wgsl"))?;
    gpu.dispatch(
        &p,
        &[tok, pos, ids, out],
        &[rows as u32, c as u32, t as u32, pos_offset as u32],
        one_d(gpu, rows * c)?,
    )
}

/// `y = x * w + bias` for one row: `x` has `k` elements, `w` is `k x n`, `bias` and `y` have `n`.
pub fn gemv(
    gpu: &Gpu,
    (x, w, bias): (&GpuBuf, &GpuBuf, &GpuBuf),
    y: &GpuBuf,
    (k, n): (usize, usize),
) -> Result<(), GpuError> {
    need("gemv x", x, k)?;
    need("gemv w", w, k * n)?;
    need("gemv bias", bias, n)?;
    need("gemv y", y, n)?;
    let p = gpu.pipeline("gemv", include_str!("shaders/gemv.wgsl"))?;
    gpu.dispatch(&p, &[x, w, bias, y], &[k as u32, n as u32], one_d(gpu, n)?)
}

/// Single-query attention over a KV cache laid out `[ctx, heads*hd]`: positions `0..len` are
/// visible. `probs` is scratch of `heads * ctx`; `out` has `heads * hd` elements.
pub fn attention_decode(
    gpu: &Gpu,
    (q, kc, vc): (&GpuBuf, &GpuBuf, &GpuBuf),
    (probs, out): (&GpuBuf, &GpuBuf),
    (heads, hd, len, ctx): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let c = heads * hd;
    if len == 0 || len > ctx {
        return Err(err(format!(
            "attention_decode: len {len} must be in 1..={ctx}"
        )));
    }
    need("attention_decode q", q, c)?;
    need("attention_decode out", out, c)?;
    need("attention_decode kc", kc, ctx * c)?;
    need("attention_decode vc", vc, ctx * c)?;
    need("attention_decode probs", probs, heads * ctx)?;
    let u = [heads as u32, hd as u32, len as u32, ctx as u32];
    let p1 = gpu.pipeline(
        "attn_dec_probs",
        include_str!("shaders/attn_dec_probs.wgsl"),
    )?;
    gpu.dispatch(&p1, &[q, kc, probs], &u, one_d(gpu, heads)?)?;
    let p2 = gpu.pipeline("attn_dec_pv", include_str!("shaders/attn_dec_pv.wgsl"))?;
    gpu.dispatch(&p2, &[probs, vc, out], &u, one_d(gpu, c)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::open_for_test;
    use crate::tolerance::{max_rel_err, EXACT, REDUCING};
    use ferrum_ref::matmul::rand_vec;
    use ferrum_ref::nn;

    fn show(name: &str, shape: &str, e: f32, tol: f32) {
        eprintln!("{name} {shape}: max rel err {e:.2e} (tol {tol:.0e})");
        assert!(e < tol, "{name} {shape}: max rel err {e} >= {tol}");
    }

    /// Scale random data so GELU/softmax see a useful range, including large magnitudes.
    fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
        rand_vec(n, seed).into_iter().map(|v| v * by).collect()
    }

    #[test]
    fn add_and_bias_match_cpu() {
        let Some(g) = open_for_test() else { return };
        for n in [1usize, 7, 64, 65, 1000] {
            let (a, b) = (rand_vec(n, 1), rand_vec(n, 2));
            let (ga, gb, gc) = (
                g.upload(&a).unwrap(),
                g.upload(&b).unwrap(),
                g.zeros(n).unwrap(),
            );
            add(&g, &ga, &gb, &gc).unwrap();
            show(
                "add",
                &format!("n={n}"),
                max_rel_err(&g.read(&gc).unwrap(), &nn::add(&a, &b)),
                EXACT,
            );
        }
        for (rows, cols) in [(1usize, 1usize), (3, 5), (17, 64), (2, 100)] {
            let (x, bias) = (rand_vec(rows * cols, 3), rand_vec(cols, 4));
            let (gx, gb, gy) = (
                g.upload(&x).unwrap(),
                g.upload(&bias).unwrap(),
                g.zeros(rows * cols).unwrap(),
            );
            add_bias(&g, &gx, &gb, &gy).unwrap();
            show(
                "add_bias",
                &format!("{rows}x{cols}"),
                max_rel_err(&g.read(&gy).unwrap(), &nn::add_bias(&x, &bias)),
                EXACT,
            );
        }
    }

    #[test]
    fn gelu_matches_cpu_including_saturation() {
        let Some(g) = open_for_test() else { return };
        let mut x = spread(1000, 5, 6.0);
        x.extend([0.0, 20.0, -20.0, 100.0, -100.0, 1e6, -1e6, 10.0, -10.0]);
        let (gx, gy) = (g.upload(&x).unwrap(), g.zeros(x.len()).unwrap());
        gelu(&g, &gx, &gy).unwrap();
        let got = g.read(&gy).unwrap();
        assert!(
            got.iter().all(|v| v.is_finite()),
            "no NaN/inf for huge inputs"
        );
        let want: Vec<f32> = x.iter().map(|&v| nn::gelu(v)).collect();
        show("gelu", "mixed", max_rel_err(&got, &want), REDUCING);
        for (i, want) in [
            (0usize, 0.0f32),
            (1, 20.0),
            (2, 0.0),
            (3, 100.0),
            (4, 0.0),
            (5, 1e6),
            (6, 0.0),
        ] {
            assert!(
                (got[1000 + i] - want).abs() <= 1e-6 * want.abs().max(1.0),
                "saturated value {i}: {}",
                got[1000 + i]
            );
        }
    }

    #[test]
    fn layernorm_matches_cpu_on_odd_shapes() {
        let Some(g) = open_for_test() else { return };
        for (rows, cols) in [
            (1usize, 1usize),
            (1, 7),
            (5, 3),
            (64, 96),
            (65, 33),
            (3, 1000),
        ] {
            let x = spread(rows * cols, 6, 3.0);
            let (gain, shift) = (rand_vec(cols, 7), rand_vec(cols, 8));
            let (gx, gg, gs, gy) = (
                g.upload(&x).unwrap(),
                g.upload(&gain).unwrap(),
                g.upload(&shift).unwrap(),
                g.zeros(rows * cols).unwrap(),
            );
            layernorm(&g, &gx, &gg, &gs, &gy, (rows, cols)).unwrap();
            let want = nn::layernorm(&x, &gain, &shift, rows, cols);
            show(
                "layernorm",
                &format!("{rows}x{cols}"),
                max_rel_err(&g.read(&gy).unwrap(), &want),
                REDUCING,
            );
        }
    }

    #[test]
    fn attention_matches_cpu() {
        let Some(g) = open_for_test() else { return };
        for shape in [
            (1usize, 1usize, 1usize, 1usize),
            (1, 1, 2, 4),
            (2, 5, 2, 3),
            (1, 17, 4, 8),
            (2, 33, 3, 5),
        ] {
            let (batch, t, heads, hd) = shape;
            let n = batch * t * heads * hd;
            let (q, k, v) = (spread(n, 1, 2.0), spread(n, 2, 2.0), rand_vec(n, 3));
            let (gq, gk, gv) = (
                g.upload(&q).unwrap(),
                g.upload(&k).unwrap(),
                g.upload(&v).unwrap(),
            );
            let (gp, go) = (g.zeros(batch * heads * t * t).unwrap(), g.zeros(n).unwrap());
            attention(&g, (&gq, &gk, &gv), (&gp, &go), shape).unwrap();
            let (want_out, want_probs) = nn::attention(&q, &k, &v, shape);
            show(
                "attention probs",
                &format!("{shape:?}"),
                max_rel_err(&g.read(&gp).unwrap(), &want_probs),
                REDUCING,
            );
            show(
                "attention out",
                &format!("{shape:?}"),
                max_rel_err(&g.read(&go).unwrap(), &want_out),
                REDUCING,
            );
        }
    }

    #[test]
    fn cross_entropy_matches_cpu() {
        let Some(g) = open_for_test() else { return };
        for (rows, vocab) in [(1usize, 2usize), (3, 5), (16, 1024), (33, 7)] {
            let logits = spread(rows * vocab, 9, 8.0);
            let targets: Vec<usize> = (0..rows).map(|i| (i * 7 + 1) % vocab).collect();
            let (gl, gt) = (
                g.upload(&logits).unwrap(),
                g.upload_indices(&targets, vocab).unwrap(),
            );
            let (gg, gr, gloss) = (
                g.zeros(rows * vocab).unwrap(),
                g.zeros(rows).unwrap(),
                g.zeros(1).unwrap(),
            );
            cross_entropy(&g, &gl, &gt, (&gg, &gr, &gloss), (rows, vocab)).unwrap();
            let (want_loss, want_grad) = nn::cross_entropy(&logits, &targets, vocab);
            show(
                "ce loss",
                &format!("{rows}x{vocab}"),
                max_rel_err(&g.read(&gloss).unwrap(), &[want_loss]),
                REDUCING,
            );
            show(
                "ce grad",
                &format!("{rows}x{vocab}"),
                max_rel_err(&g.read(&gg).unwrap(), &want_grad),
                REDUCING,
            );
        }
    }

    #[test]
    fn score_matches_cpu_including_ties() {
        let Some(g) = open_for_test() else { return };
        let (rows, vocab) = (5usize, 6usize);
        let mut logits = spread(rows * vocab, 11, 4.0);
        logits[vocab..2 * vocab].fill(1.5); // a fully tied row: first index is the argmax
        let targets = vec![0usize, 0, 3, 5, 1];
        let (gl, gt, go) = (
            g.upload(&logits).unwrap(),
            g.upload_indices(&targets, vocab).unwrap(),
            g.zeros(2 * rows).unwrap(),
        );
        score(&g, &gl, &gt, &go, (rows, vocab)).unwrap();
        let got = g.read(&go).unwrap();
        let want = nn::score(&logits, &targets, vocab);
        for (i, (lp, hit)) in want.iter().enumerate() {
            assert!(
                (got[2 * i] - lp).abs() < 1e-4 * lp.abs().max(1.0),
                "row {i}: {} vs {lp}",
                got[2 * i]
            );
            assert_eq!(got[2 * i + 1] == 1.0, *hit, "row {i} argmax flag");
        }
        assert_eq!(got[3], 1.0, "tied row with target 0 counts as argmax");
    }

    #[test]
    fn embed_matches_cpu_and_rejects_bad_ids() {
        let Some(g) = open_for_test() else { return };
        let (vocab, c, t) = (11usize, 6usize, 4usize);
        let (tok, pos) = (rand_vec(vocab * c, 12), rand_vec(t * c, 13));
        let ids = vec![0usize, 10, 3, 3, 7, 1, 2]; // 7 rows, two sequences' worth of positions
        let (gt, gp, gi) = (
            g.upload(&tok).unwrap(),
            g.upload(&pos).unwrap(),
            g.upload_indices(&ids, vocab).unwrap(),
        );
        let go = g.zeros(ids.len() * c).unwrap();
        embed(&g, (&gt, &gp, &gi), &go, (ids.len(), c, t)).unwrap();
        show(
            "embed",
            "7x6",
            max_rel_err(&g.read(&go).unwrap(), &nn::embed(&tok, &pos, &ids, t, c)),
            EXACT,
        );
        let e = g
            .upload_indices(&[3, 11], vocab)
            .expect_err("id 11 is out of range");
        assert!(e.to_string().contains("11"));
    }

    #[test]
    fn shape_mismatches_are_errors_not_dispatches() {
        let Some(g) = open_for_test() else { return };
        let (a, b, c) = (
            g.zeros(4).unwrap(),
            g.zeros(5).unwrap(),
            g.zeros(4).unwrap(),
        );
        assert!(add(&g, &a, &b, &c)
            .unwrap_err()
            .to_string()
            .contains("expected 4"));
        assert!(add_bias(&g, &a, &g.zeros(3).unwrap(), &c).is_err());
        assert!(add_bias(&g, &a, &g.zeros(0).unwrap(), &c).is_err());
        assert!(layernorm(
            &g,
            &a,
            &g.zeros(2).unwrap(),
            &g.zeros(2).unwrap(),
            &c,
            (2, 3)
        )
        .is_err());
        let (l, t) = (g.zeros(6).unwrap(), g.upload_indices(&[0, 1], 3).unwrap());
        let one = g.zeros(1).unwrap();
        assert!(cross_entropy(
            &g,
            &l,
            &t,
            (&g.zeros(6).unwrap(), &g.zeros(2).unwrap(), &one),
            (0, 3)
        )
        .is_err());
        assert!(embed(
            &g,
            (&g.zeros(4).unwrap(), &g.zeros(2).unwrap(), &t),
            &g.zeros(8).unwrap(),
            (2, 4, 4)
        )
        .is_err());
    }

    #[test]
    fn gemv_matches_cpu_linear() {
        let Some(g) = open_for_test() else { return };
        for (k, n) in [(1usize, 1usize), (7, 5), (96, 64), (65, 130), (320, 96)] {
            let (x, w, b) = (rand_vec(k, 1), rand_vec(k * n, 2), rand_vec(n, 3));
            let want = nn::add_bias(&ferrum_ref::matmul::matmul(&x, &w, 1, k, n).unwrap(), &b);
            let (gx, gw, gb, gy) = (
                g.upload(&x).unwrap(),
                g.upload(&w).unwrap(),
                g.upload(&b).unwrap(),
                g.zeros(n).unwrap(),
            );
            gemv(&g, (&gx, &gw, &gb), &gy, (k, n)).unwrap();
            show(
                "gemv",
                &format!("{k}x{n}"),
                max_rel_err(&g.read(&gy).unwrap(), &want),
                REDUCING,
            );
        }
        let z = g.zeros(2).unwrap();
        assert!(gemv(&g, (&z, &g.zeros(3).unwrap(), &z,), &z, (2, 2)).is_err());
    }

    #[test]
    fn attention_decode_matches_the_last_row_of_full_attention() {
        let Some(g) = open_for_test() else { return };
        let (heads, hd, t, ctx) = (3usize, 5usize, 9usize, 12usize);
        let c = heads * hd;
        let (q, k, v) = (
            spread(t * c, 31, 2.0),
            spread(t * c, 32, 2.0),
            rand_vec(t * c, 33),
        );
        let (want_out, _) = nn::attention(&q, &k, &v, (1, t, heads, hd));
        for len in [1usize, 4, t] {
            // Cache holds positions 0..len; the query is position len-1.
            let mut kc = vec![0.0; ctx * c];
            let mut vc = vec![0.0; ctx * c];
            kc[..len * c].copy_from_slice(&k[..len * c]);
            vc[..len * c].copy_from_slice(&v[..len * c]);
            let (gq, gk, gv) = (
                g.upload(&q[(len - 1) * c..len * c]).unwrap(),
                g.upload(&kc).unwrap(),
                g.upload(&vc).unwrap(),
            );
            let (gp, go) = (g.zeros(heads * ctx).unwrap(), g.zeros(c).unwrap());
            attention_decode(&g, (&gq, &gk, &gv), (&gp, &go), (heads, hd, len, ctx)).unwrap();
            let want = &want_out[(len - 1) * c..len * c];
            show(
                "attention_decode",
                &format!("len={len}"),
                max_rel_err(&g.read(&go).unwrap(), want),
                REDUCING,
            );
        }
        let z = g.zeros(c).unwrap();
        let big = g.zeros(ctx * c).unwrap();
        let pr = g.zeros(heads * ctx).unwrap();
        assert!(attention_decode(&g, (&z, &big, &big), (&pr, &z), (heads, hd, 0, ctx)).is_err());
        assert!(
            attention_decode(&g, (&z, &big, &big), (&pr, &z), (heads, hd, ctx + 1, ctx)).is_err()
        );
    }

    #[test]
    fn embed_at_uses_the_absolute_position() {
        let Some(g) = open_for_test() else { return };
        let (vocab, c, t) = (6usize, 4usize, 5usize);
        let (tok, pos) = (rand_vec(vocab * c, 41), rand_vec(t * c, 42));
        for p in [0usize, 3, 4] {
            let gi = g.upload_indices(&[2], vocab).unwrap();
            let (gt, gp, go) = (
                g.upload(&tok).unwrap(),
                g.upload(&pos).unwrap(),
                g.zeros(c).unwrap(),
            );
            embed_at(&g, (&gt, &gp, &gi), &go, (1, c, t), p).unwrap();
            let want: Vec<f32> = (0..c).map(|d| tok[2 * c + d] + pos[p * c + d]).collect();
            show(
                "embed_at",
                &format!("pos={p}"),
                max_rel_err(&g.read(&go).unwrap(), &want),
                EXACT,
            );
        }
    }

    #[test]
    fn repeat_runs_are_bit_identical() {
        let Some(g) = open_for_test() else { return };
        let (rows, vocab) = (20usize, 50usize);
        let logits = spread(rows * vocab, 21, 5.0);
        let targets: Vec<usize> = (0..rows).map(|i| i % vocab).collect();
        let run = || {
            let (gl, gt) = (
                g.upload(&logits).unwrap(),
                g.upload_indices(&targets, vocab).unwrap(),
            );
            let (gg, gr, gloss) = (
                g.zeros(rows * vocab).unwrap(),
                g.zeros(rows).unwrap(),
                g.zeros(1).unwrap(),
            );
            cross_entropy(&g, &gl, &gt, (&gg, &gr, &gloss), (rows, vocab)).unwrap();
            (g.read(&gg).unwrap(), g.read(&gloss).unwrap())
        };
        assert_eq!(run(), run());
    }
}
