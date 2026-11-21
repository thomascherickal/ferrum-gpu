use super::*;

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
