use super::*;

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

pub(crate) struct Out {
    pub(crate) b: GgufBuilder,
    pub(crate) quant: Quant,
}
