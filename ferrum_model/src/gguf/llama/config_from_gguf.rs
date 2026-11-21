use super::*;

/// The model shape from the `{arch}.*` metadata and the embedding tensor, before validation.
pub(super) fn read_config(g: &Gguf, arch: &str, rope_type: RopeType) -> Result<Config, GgufError> {
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
    Ok(cfg)
}
