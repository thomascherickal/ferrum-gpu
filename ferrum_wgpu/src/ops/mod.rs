//! Forward kernels of the GPT-2 stack. Every function records one or more dispatches on
//! resident buffers and does not wait; read results with [`Gpu::read`].
//! ponytail: 1-D kernels cap at 65,535 workgroups x 64 invocations (4.1M elements) per call;
//! add a 2-D split when a tensor needs more.

/// `pipe!(gpu, "group/name")`: the cached compute pipeline of `shaders/group/name.wgsl`. A macro
/// (not a function) because `include_str!` needs a literal path; it keeps every call site on one
/// short line.
macro_rules! pipe {
    ($gpu:expr, $name:literal) => {
        $gpu.pipeline($name, include_str!(concat!("../shaders/", $name, ".wgsl")))
    };
}

pub mod bwd;
pub mod dropout;
pub mod llama;
pub mod llama_bwd;
pub mod matmul;
pub mod opt;
pub mod qat;
pub mod quant;

use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;

pub(crate) fn err(msg: String) -> GpuError {
    GpuError(msg)
}

pub(crate) fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
    if b.len() == want {
        Ok(())
    } else {
        Err(err(format!(
            "{name}: buffer holds {} elements, expected {want}",
            b.len()
        )))
    }
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
    let p = pipe!(gpu, "fwd/add")?;
    gpu.dispatch_1d(&p, &[a, b, c], &[n as u32], n)
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
    let p = pipe!(gpu, "fwd/bias_add")?;
    gpu.dispatch_1d(
        &p,
        &[x, bias, y],
        &[x.len() as u32, bias.len() as u32],
        x.len(),
    )
}

/// `y = gelu(x)` (tanh approximation).
pub fn gelu(gpu: &Gpu, x: &GpuBuf, y: &GpuBuf) -> Result<(), GpuError> {
    need("gelu y", y, x.len())?;
    let p = pipe!(gpu, "fwd/gelu")?;
    gpu.dispatch_1d(&p, &[x, y], &[x.len() as u32], x.len())
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
    let p = pipe!(gpu, "fwd/layernorm")?;
    gpu.dispatch_1d(&p, &[x, gain, shift, y], &[rows as u32, cols as u32], rows)
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
    // Scores are computed for every (row, key) cell in parallel, then softmaxed per row.
    let p0 = pipe!(gpu, "fwd/attn_scores")?;
    gpu.dispatch_1d(&p0, &[q, k, probs], &u, batch * heads * t * t)?;
    let p1 = pipe!(gpu, "fwd/attn_softmax")?;
    gpu.dispatch_1d(&p1, &[probs], &u, batch * heads * t)?;
    let p2 = pipe!(gpu, "fwd/attn_pv")?;
    gpu.dispatch_1d(&p2, &[probs, v, out], &u, batch * t * c)
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
    let p1 = pipe!(gpu, "fwd/ce_rows")?;
    gpu.dispatch_1d(
        &p1,
        &[logits, targets, grad, row_loss],
        &[rows as u32, vocab as u32],
        rows,
    )?;
    let p2 = pipe!(gpu, "fwd/mean_loss")?;
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
    let p = pipe!(gpu, "fwd/score")?;
    gpu.dispatch_1d(
        &p,
        &[logits, targets, out],
        &[rows as u32, vocab as u32],
        rows,
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
    let p = pipe!(gpu, "fwd/embed")?;
    gpu.dispatch_1d(
        &p,
        &[tok, pos, ids, out],
        &[rows as u32, c as u32, t as u32, pos_offset as u32],
        rows * c,
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
    let p = pipe!(gpu, "fwd/gemv")?;
    gpu.dispatch_1d(&p, &[x, w, bias, y], &[k as u32, n as u32], n)
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
    let p1 = pipe!(gpu, "fwd/attn_dec_probs")?;
    gpu.dispatch_1d(&p1, &[q, kc, probs], &u, heads)?;
    let p2 = pipe!(gpu, "fwd/attn_dec_pv")?;
    gpu.dispatch_1d(&p2, &[probs, vc, out], &u, c)
}
#[cfg(test)]
mod tests;
