// Provenance: tensor names, the QKV stacking and the transposition convention follow
// ferrum-cpu `ferrum_core/src/loader/gpt2.rs` and `loader/encode.rs` (commit 56a9d3a), which
// write llama.cpp's `gpt2` layout. The code is our own.
//! `gpt2` GGUF <-> [`Gpt2`].
//!
//! GGUF stores each matrix with GGML dim 0 (contiguous) = the input dimension, so a stored
//! matrix is `[out, in]` row-major; our weights are `[in, out]`, hence a transpose each way.
//! `attn_qkv.weight` stacks the Q, K, V matrices (each transposed) along the output axis.
//! A missing `output.weight` means the head is tied to `token_embd.weight`.

use ferrum_gguf::quant::encode;
use ferrum_gguf::{Gguf, GgufBuilder, GgufError, Value};
use ferrum_ref::gpt2::{Block, Config, Gpt2};
use ferrum_ref::nn::LN_EPS;

pub use super::quant::Quant;

fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<(), GgufError> {
    if got == want {
        Ok(())
    } else {
        Err(err(format!(
            "tensor '{name}' has {got} elements, expected {want}"
        )))
    }
}

/// Read a tensor, check its element count, return its f32 data.
fn load(g: &Gguf, name: &str, want: usize) -> Result<Vec<f32>, GgufError> {
    let v = g.tensor_f32(name)?;
    expect_len(name, v.len(), want)?;
    Ok(v)
}

/// Load a stored `[out, in]` matrix as our `[in, out]`.
fn load_matrix(g: &Gguf, name: &str, inp: usize, out: usize) -> Result<Vec<f32>, GgufError> {
    Ok(transpose(&load(g, name, inp * out)?, out, inp))
}

/// Build a [`Gpt2`] from a `gpt2`-architecture GGUF.
pub fn from_gguf(g: &Gguf) -> Result<Gpt2, GgufError> {
    let arch = g.string("general.architecture")?;
    if arch != "gpt2" {
        return Err(err(format!(
            "architecture is '{arch}', only 'gpt2' is supported"
        )));
    }
    let eps = g.float("gpt2.attention.layer_norm_epsilon")?;
    if (eps - LN_EPS).abs() > 1e-9 {
        return Err(err(format!(
            "layer_norm_epsilon is {eps}; the kernels use {LN_EPS}"
        )));
    }
    let k = |s: &str| g.uint(&format!("gpt2.{s}"));
    let (d, ctx) = (k("embedding_length")?, k("context_length")?);
    let tok_info = g
        .tensor("token_embd.weight")
        .ok_or_else(|| err("missing tensor 'token_embd.weight'"))?;
    if tok_info.dims.len() != 2 || tok_info.dims[0] != d as u64 {
        return Err(err("token_embd.weight must be [embedding_length, vocab]"));
    }
    let cfg = Config {
        vocab: tok_info.dims[1] as usize,
        ctx,
        dim: d,
        heads: k("attention.head_count")?,
        layers: k("block_count")?,
        hidden: k("feed_forward_length")?,
    };
    cfg.validate().map_err(err)?;
    // Header numbers size every allocation below, so bound them by the file's own tensors:
    // the embedding tensor was already proven to fit inside the file by the parser.
    let tok = load(g, "token_embd.weight", cfg.vocab * d)?;
    let pos = load(g, "position_embd.weight", ctx * d)?;
    let mut blocks = Vec::with_capacity(cfg.layers);
    for i in 0..cfg.layers {
        let n = |s: &str| format!("blk.{i}.{s}");
        let qkv = load(g, &n("attn_qkv.weight"), 3 * d * d)?;
        let qkv_b = load(g, &n("attn_qkv.bias"), 3 * d)?;
        let part = |j: usize| transpose(&qkv[j * d * d..(j + 1) * d * d], d, d);
        blocks.push(Block {
            ln1_g: load(g, &n("attn_norm.weight"), d)?,
            ln1_b: load(g, &n("attn_norm.bias"), d)?,
            wq: part(0),
            bq: qkv_b[..d].to_vec(),
            wk: part(1),
            bk: qkv_b[d..2 * d].to_vec(),
            wv: part(2),
            bv: qkv_b[2 * d..].to_vec(),
            wo: load_matrix(g, &n("attn_output.weight"), d, d)?,
            bo: load(g, &n("attn_output.bias"), d)?,
            ln2_g: load(g, &n("ffn_norm.weight"), d)?,
            ln2_b: load(g, &n("ffn_norm.bias"), d)?,
            w1: load_matrix(g, &n("ffn_up.weight"), d, cfg.hidden)?,
            b1: load(g, &n("ffn_up.bias"), cfg.hidden)?,
            w2: load_matrix(g, &n("ffn_down.weight"), cfg.hidden, d)?,
            b2: load(g, &n("ffn_down.bias"), d)?,
        });
    }
    let head = if g.tensor("output.weight").is_some() {
        load_matrix(g, "output.weight", d, cfg.vocab)?
    } else {
        transpose(&tok, cfg.vocab, d) // tied: head[in][out] = tok[out][in]
    };
    Ok(Gpt2 {
        cfg,
        lnf_g: load(g, "output_norm.weight", d)?,
        lnf_b: load(g, "output_norm.bias", d)?,
        tok,
        pos,
        blocks,
        head,
    })
}

struct Out {
    b: GgufBuilder,
    quant: Quant,
}

impl Out {
    fn vec(&mut self, name: &str, data: &[f32]) {
        self.b.tensor_f32(name, &[data.len() as u64], data);
    }

    /// `rows` GGML rows of `ne0` values, quantised per the precision when `ne0 % 32 == 0`.
    fn rows(&mut self, name: &str, data: &[f32], rows: usize, ne0: usize) {
        let ty = self.quant.tensor_type(ne0);
        let bytes = encode(ty, data).expect("block-aligned by the check above");
        self.b
            .tensor_raw(name, &[ne0 as u64, rows as u64], ty, bytes);
    }

    /// Our `[in, out]` matrix, stored transposed.
    fn matrix(&mut self, name: &str, w: &[f32], inp: usize, out: usize) {
        self.rows(name, &transpose(w, inp, out), out, inp);
    }
}

/// Serialise as a `gpt2` GGUF with no tokenizer (byte-level token ids).
pub fn to_gguf(m: &Gpt2, quant: Quant) -> Vec<u8> {
    to_gguf_with(m, quant, None)
}

/// The generic `ferrum.metadata` JSON ferrum-cpu reads from every model file. Its shape follows a
/// ferrum-cpu `gpt2` file (fixtures/bpe/model.gguf); the fields a language model does not use are
/// the same constants.
fn ferrum_metadata(c: &Config) -> String {
    let names: Vec<String> = (0..c.ctx).map(|i| format!("\"c_{i}\"")).collect();
    let range = format!("[0.000000,{:.6}]", c.vocab as f64);
    let ranges = vec![range.clone(); c.ctx].join(",");
    format!(
        "{{\"dataset_name\":\"ferrum-gpu Transformer\",\"task\":\"transformer_slm\",\"feature_names\":[{}],\"feature_ranges\":[{ranges}],\"class_names\":[],\"target_name\":\"next_char\",\"target_range\":{range},\"input_dim\":{},\"output_dim\":{},\"tokenizer_state\":\"\"}}",
        names.join(","),
        c.ctx,
        c.vocab
    )
}

/// Serialise as a `gpt2` GGUF. The tied head is omitted when it equals the transposed
/// embedding. With a tokenizer the file carries the `tokenizer.ggml.*` vocabulary (so llama.cpp
/// tokenizes the same way) and the `ferrum.*` keys ferrum-cpu needs to load it; the tokenizer's
/// vocabulary size must equal the model's.
pub fn to_gguf_with(m: &Gpt2, quant: Quant, tok: Option<&ferrum_tok::ByteBpeTokenizer>) -> Vec<u8> {
    let (c, d) = (&m.cfg, m.cfg.dim);
    let mut o = Out {
        b: GgufBuilder::new(),
        quant,
    };
    o.b.meta("general.architecture", Value::Str("gpt2".into()));
    o.b.meta("general.name", Value::Str("ferrum-gpu Transformer".into()));
    o.b.meta("general.quantization_version", Value::U32(2));
    o.b.meta("ferrum.format", Value::U32(1));
    o.b.meta("ferrum.quant", Value::Str(quant.tag().into()));
    o.b.meta("ferrum.normalizer", Value::Str(String::new()));
    o.b.meta("ferrum.metadata", Value::Str(ferrum_metadata(c)));
    if let Some(t) = tok {
        crate::tokenizer_gguf::put_vocab(&mut o.b, t);
    }
    for (key, v) in [
        ("context_length", c.ctx),
        ("embedding_length", d),
        ("block_count", c.layers),
        ("feed_forward_length", c.hidden),
        ("attention.head_count", c.heads),
    ] {
        o.b.meta(&format!("gpt2.{key}"), Value::U32(v as u32));
    }
    o.b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
    o.rows("token_embd.weight", &m.tok, c.vocab, d);
    o.b.tensor_f32("position_embd.weight", &[d as u64, c.ctx as u64], &m.pos);
    for (i, b) in m.blocks.iter().enumerate() {
        let n = |s: &str| format!("blk.{i}.{s}");
        o.vec(&n("attn_norm.weight"), &b.ln1_g);
        o.vec(&n("attn_norm.bias"), &b.ln1_b);
        let qkv: Vec<f32> = [&b.wq, &b.wk, &b.wv]
            .iter()
            .flat_map(|w| transpose(w, d, d))
            .collect();
        o.rows(&n("attn_qkv.weight"), &qkv, 3 * d, d);
        o.vec(&n("attn_qkv.bias"), &[&b.bq[..], &b.bk, &b.bv].concat());
        o.matrix(&n("attn_output.weight"), &b.wo, d, d);
        o.vec(&n("attn_output.bias"), &b.bo);
        o.vec(&n("ffn_norm.weight"), &b.ln2_g);
        o.vec(&n("ffn_norm.bias"), &b.ln2_b);
        o.matrix(&n("ffn_up.weight"), &b.w1, d, c.hidden);
        o.vec(&n("ffn_up.bias"), &b.b1);
        o.matrix(&n("ffn_down.weight"), &b.w2, c.hidden, d);
        o.vec(&n("ffn_down.bias"), &b.b2);
    }
    o.vec("output_norm.weight", &m.lnf_g);
    o.vec("output_norm.bias", &m.lnf_b);
    let head_rows = transpose(&m.head, d, c.vocab);
    if head_rows != m.tok {
        o.rows("output.weight", &head_rows, c.vocab, d);
    }
    o.b.into_bytes()
}
#[cfg(test)]
mod tests;
