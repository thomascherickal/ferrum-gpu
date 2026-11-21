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
    let cfg = read_config(g, &arch, rope_type)?;
    cfg.validate().map_err(err)?;
    let rope = rope_scaling(g, &arch, cfg.rope_dim)?;
    let d = cfg.dim;
    let tok = load(g, "token_embd.weight", &[d, cfg.vocab])?;
    let blocks = (0..cfg.layers)
        .map(|i| load_block(g, i, &cfg))
        .collect::<Result<Vec<_>, _>>()?;
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
