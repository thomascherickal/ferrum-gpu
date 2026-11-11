// Provenance: tensor names, metadata keys, the `[in, out]` shape checks, the tied-head rule, the
// rope-scaling policy (none / linear / `rope_freqs.weight`, everything else refused) and the
// arch -> RoPE pairing rule (llama = NORM, qwen2 = NEOX) follow ferrum-cpu
// `ferrum_core/src/gguf/llama.rs` (commit 56a9d3a), which follows llama.cpp. The code is our own.
//! `llama` / `qwen2` GGUF <-> [`Llama`].
//!
//! GGUF stores each matrix with GGML dim 0 (contiguous) = the input dimension, so a stored
//! matrix is `[out, in]` row-major; our weights are `[in, out]`, hence a transpose each way. A
//! missing `output.weight` means the head is tied to `token_embd.weight`. Biases are accepted on
//! `attn_q`, `attn_k`, `attn_v` only; a bias anywhere else is refused rather than dropped.

#[cfg(test)]
mod tests;

use crate::gpt2_gguf::Quant;
use ferrum_gguf::quant::encode;
use ferrum_gguf::{Gguf, GgufBuilder, GgufError, Value, GGML_F32, GGML_Q4_0, GGML_Q8_0, QK};
use ferrum_ref::llama::{transpose, Block, Config, Llama, RopeScaling, RopeType};

fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

fn opt_uint(g: &Gguf, key: &str) -> Result<Option<usize>, GgufError> {
    g.get(key).map(|_| g.uint(key)).transpose()
}

fn opt_float(g: &Gguf, key: &str) -> Result<Option<f32>, GgufError> {
    g.get(key).map(|_| g.float(key)).transpose()
}

/// Read a tensor that must have exactly the GGML dims `dims`, as f32.
fn load(g: &Gguf, name: &str, dims: &[usize]) -> Result<Vec<f32>, GgufError> {
    let t = g
        .tensor(name)
        .ok_or_else(|| err(format!("missing tensor '{name}'")))?;
    let want: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
    if t.dims != want {
        return Err(err(format!(
            "tensor '{name}' has dims {:?}, expected {want:?} from the metadata",
            t.dims
        )));
    }
    g.tensor_f32(name)
}

/// A stored `[out, in]` matrix as our `[in, out]`.
fn load_matrix(g: &Gguf, name: &str, inp: usize, out: usize) -> Result<Vec<f32>, GgufError> {
    Ok(transpose(&load(g, name, &[inp, out])?, out, inp))
}

/// An optional bias vector (empty when the file has none).
fn load_bias(g: &Gguf, name: &str, n: usize) -> Result<Vec<f32>, GgufError> {
    if g.tensor(name).is_some() {
        load(g, name, &[n])
    } else {
        Ok(Vec::new())
    }
}

/// RoPE scaling as llama.cpp reads it. `yarn`, `longrope` and a non-1 `attn_factor` are refused:
/// running them unscaled would be silently wrong.
fn rope_scaling(g: &Gguf, arch: &str, rope_dim: usize) -> Result<RopeScaling, GgufError> {
    let key = |k: &str| format!("{arch}.rope.{k}");
    let factor = match opt_float(g, &key("scaling.factor"))? {
        Some(f) => Some(f),
        None => opt_float(g, &key("scale_linear"))?,
    };
    let ty = match g.get(&key("scaling.type")) {
        Some(_) => g.string(&key("scaling.type"))?,
        None => "linear",
    };
    let factor = factor.filter(|&f| f != 0.0).unwrap_or(1.0);
    let linear_factor = match ty {
        "linear" => factor,
        "none" if factor == 1.0 => 1.0,
        "none" => {
            return Err(err(format!(
                "rope scaling type is \"none\" but the factor is {factor}"
            )))
        }
        other => {
            return Err(err(format!(
            "rope scaling type '{other}' is not supported (none, linear and rope_freqs.weight are)"
        )))
        }
    };
    let attn = opt_float(g, &key("scaling.attn_factor"))?.unwrap_or(1.0);
    if attn != 1.0 {
        return Err(err(format!(
            "rope scaling attn_factor {attn} is not supported"
        )));
    }
    let freq_factors = if g.tensor("rope_freqs.weight").is_some() {
        load(g, "rope_freqs.weight", &[rope_dim / 2])?
    } else {
        Vec::new()
    };
    let s = RopeScaling {
        freq_factors,
        linear_factor,
    };
    s.check(rope_dim).map_err(err)?;
    Ok(s)
}

/// Build a [`Llama`] from a `llama`- or `qwen2`-architecture GGUF.
pub fn from_gguf(g: &Gguf) -> Result<Llama, GgufError> {
    let arch = g.string("general.architecture")?.to_string();
    let rope_type = match arch.as_str() {
        "llama" => RopeType::Norm,
        "qwen2" => RopeType::Neox,
        other => {
            return Err(err(format!(
                "architecture is '{other}', only 'llama' and 'qwen2' are supported"
            )))
        }
    };
    let k = |s: &str| g.uint(&format!("{arch}.{s}"));
    let ko = |s: &str| opt_uint(g, &format!("{arch}.{s}"));
    let dim = k("embedding_length")?;
    let heads = k("attention.head_count")?;
    let kv_heads = ko("attention.head_count_kv")?.unwrap_or(heads);
    // Each layer needs nine tensors, so the tensor count bounds the layer count before it sizes
    // anything; the embedding tensor below bounds vocab and dim the same way.
    let layers = k("block_count")?;
    if layers > g.tensors.len() {
        return Err(err(format!(
            "block_count {layers} exceeds the file's {} tensors",
            g.tensors.len()
        )));
    }
    let head_dim = match ko("attention.key_length")? {
        Some(h) => h,
        None if heads > 0 => dim / heads,
        None => 0,
    };
    let cfg_eps = opt_float(g, &format!("{arch}.attention.layer_norm_rms_epsilon"))?;
    let emb = g
        .tensor("token_embd.weight")
        .ok_or_else(|| err("missing tensor 'token_embd.weight'"))?;
    if emb.dims.len() != 2 || emb.dims[0] != dim as u64 {
        return Err(err("token_embd.weight must be [embedding_length, vocab]"));
    }
    let cfg = Config {
        vocab: emb.dims[1] as usize,
        ctx: ko("context_length")?.unwrap_or(2048),
        dim,
        heads,
        kv_heads,
        head_dim,
        layers,
        hidden: k("feed_forward_length")?,
        rope_dim: ko("rope.dimension_count")?.unwrap_or(head_dim),
        rope_base: opt_float(g, &format!("{arch}.rope.freq_base"))?.unwrap_or(10000.0),
        rope_type,
        eps: cfg_eps.unwrap_or(1e-5),
    };
    cfg.validate().map_err(err)?;
    let rope = rope_scaling(g, &arch, cfg.rope_dim)?;
    let (d, h, qd, kvd) = (dim, cfg.hidden, cfg.q_dim(), cfg.kv_dim());
    let tok = load(g, "token_embd.weight", &[d, cfg.vocab])?;
    let mut blocks = Vec::with_capacity(layers);
    for i in 0..layers {
        let n = |s: &str| format!("blk.{i}.{s}");
        for stem in ["attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
            if g.tensor(&n(&format!("{stem}.bias"))).is_some() {
                return Err(err(format!(
                    "tensor '{}' is not supported: biases are loaded only for attn_q/k/v",
                    n(&format!("{stem}.bias"))
                )));
            }
        }
        blocks.push(Block {
            attn_norm: load(g, &n("attn_norm.weight"), &[d])?,
            wq: load_matrix(g, &n("attn_q.weight"), d, qd)?,
            bq: load_bias(g, &n("attn_q.bias"), qd)?,
            wk: load_matrix(g, &n("attn_k.weight"), d, kvd)?,
            bk: load_bias(g, &n("attn_k.bias"), kvd)?,
            wv: load_matrix(g, &n("attn_v.weight"), d, kvd)?,
            bv: load_bias(g, &n("attn_v.bias"), kvd)?,
            wo: load_matrix(g, &n("attn_output.weight"), qd, d)?,
            ffn_norm: load(g, &n("ffn_norm.weight"), &[d])?,
            gate: load_matrix(g, &n("ffn_gate.weight"), d, h)?,
            up: load_matrix(g, &n("ffn_up.weight"), d, h)?,
            down: load_matrix(g, &n("ffn_down.weight"), h, d)?,
        });
    }
    if g.tensor("output.bias").is_some() {
        return Err(err("tensor 'output.bias' is not supported"));
    }
    let tied = g.tensor("output.weight").is_none();
    let head = if tied {
        transpose(&tok, cfg.vocab, d)
    } else {
        load_matrix(g, "output.weight", d, cfg.vocab)?
    };
    Ok(Llama {
        cfg,
        rope,
        out_norm: load(g, "output_norm.weight", &[d])?,
        tok,
        blocks,
        head,
        tied,
    })
}

struct Out {
    b: GgufBuilder,
    quant: Quant,
    skip: Vec<String>,
}

impl Out {
    fn vec(&mut self, name: &str, data: &[f32]) {
        if !self.skip.iter().any(|s| s == name) {
            self.b.tensor_f32(name, &[data.len() as u64], data);
        }
    }

    /// `rows` GGML rows of `ne0` values, quantised per the precision when `ne0 % 32 == 0`.
    fn rows(&mut self, name: &str, data: &[f32], rows: usize, ne0: usize) {
        if self.skip.iter().any(|s| s == name) {
            return;
        }
        let ty = match self.quant {
            Quant::Q8_0 if ne0.is_multiple_of(QK) => GGML_Q8_0,
            Quant::Q4_0 if ne0.is_multiple_of(QK) => GGML_Q4_0,
            _ => GGML_F32,
        };
        let bytes = encode(ty, data).expect("block-aligned by the check above");
        self.b
            .tensor_raw(name, &[ne0 as u64, rows as u64], ty, bytes);
    }

    /// Our `[in, out]` matrix, stored transposed.
    fn matrix(&mut self, name: &str, w: &[f32], inp: usize, out: usize) {
        self.rows(name, &transpose(w, inp, out), out, inp);
    }

    fn bias(&mut self, name: &str, data: &[f32]) {
        if !data.is_empty() {
            self.vec(name, data);
        }
    }
}

/// Metadata entries in file order.
pub(crate) type Meta = Vec<(String, Value)>;

/// The metadata of `m`, in file order.
fn metadata(m: &Llama) -> Meta {
    let c = &m.cfg;
    let arch = match c.rope_type {
        RopeType::Norm => "llama",
        RopeType::Neox => "qwen2",
    };
    let u = |v: usize| Value::U32(v as u32);
    let mut kv = vec![
        ("general.architecture".to_string(), Value::Str(arch.into())),
        (format!("{arch}.context_length"), u(c.ctx)),
        (format!("{arch}.embedding_length"), u(c.dim)),
        (format!("{arch}.block_count"), u(c.layers)),
        (format!("{arch}.feed_forward_length"), u(c.hidden)),
        (format!("{arch}.attention.head_count"), u(c.heads)),
        (format!("{arch}.attention.head_count_kv"), u(c.kv_heads)),
        (format!("{arch}.attention.key_length"), u(c.head_dim)),
        (format!("{arch}.rope.dimension_count"), u(c.rope_dim)),
        (format!("{arch}.rope.freq_base"), Value::F32(c.rope_base)),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            Value::F32(c.eps),
        ),
    ];
    if m.rope.linear_factor != 1.0 {
        kv.push((
            format!("{arch}.rope.scaling.type"),
            Value::Str("linear".into()),
        ));
        kv.push((
            format!("{arch}.rope.scaling.factor"),
            Value::F32(m.rope.linear_factor),
        ));
    }
    kv
}

/// Serialise `m`, applying `edit` to the metadata and omitting the tensors named in `skip`
/// (the hostile-file tests use both; [`to_gguf`] uses neither).
pub(crate) fn write(m: &Llama, quant: Quant, edit: &dyn Fn(&mut Meta), skip: &[&str]) -> Vec<u8> {
    let (c, d) = (&m.cfg, m.cfg.dim);
    let mut o = Out {
        b: GgufBuilder::new(),
        quant,
        skip: skip.iter().map(|s| s.to_string()).collect(),
    };
    let mut kv = metadata(m);
    edit(&mut kv);
    kv.into_iter().for_each(|(k, v)| o.b.meta(&k, v));
    o.rows("token_embd.weight", &m.tok, c.vocab, d);
    for (i, b) in m.blocks.iter().enumerate() {
        let n = |s: &str| format!("blk.{i}.{s}");
        o.vec(&n("attn_norm.weight"), &b.attn_norm);
        o.matrix(&n("attn_q.weight"), &b.wq, d, c.q_dim());
        o.bias(&n("attn_q.bias"), &b.bq);
        o.matrix(&n("attn_k.weight"), &b.wk, d, c.kv_dim());
        o.bias(&n("attn_k.bias"), &b.bk);
        o.matrix(&n("attn_v.weight"), &b.wv, d, c.kv_dim());
        o.bias(&n("attn_v.bias"), &b.bv);
        o.matrix(&n("attn_output.weight"), &b.wo, c.q_dim(), d);
        o.vec(&n("ffn_norm.weight"), &b.ffn_norm);
        o.matrix(&n("ffn_gate.weight"), &b.gate, d, c.hidden);
        o.matrix(&n("ffn_up.weight"), &b.up, d, c.hidden);
        o.matrix(&n("ffn_down.weight"), &b.down, c.hidden, d);
    }
    o.vec("output_norm.weight", &m.out_norm);
    if !m.tied {
        o.matrix("output.weight", &m.head, d, c.vocab);
    }
    if !m.rope.freq_factors.is_empty() {
        o.vec("rope_freqs.weight", &m.rope.freq_factors);
    }
    o.b.into_bytes()
}

/// Serialise as a `llama` (NORM RoPE) or `qwen2` (NEOX RoPE) GGUF. A tied model omits
/// `output.weight`. No tokenizer metadata is written (Phase 6).
pub fn to_gguf(m: &Llama, quant: Quant) -> Vec<u8> {
    write(m, quant, &|_| {}, &[])
}
