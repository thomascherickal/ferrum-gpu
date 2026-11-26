use super::*;

/// Tensors a checkpoint holds for `params` parameters: the value and both Adam moments.
pub(super) fn tensors_for(params: usize) -> Result<usize, GgufError> {
    params
        .checked_mul(3)
        .ok_or_else(|| err("tensor count overflows"))
}

/// Parse a checkpoint. Every size is derived from the config with checked arithmetic and
/// compared with the file's own tensor directory before any model-sized allocation.
pub fn from_bytes(bytes: Vec<u8>) -> Result<Checkpoint, GgufError> {
    let g = Gguf::parse(bytes)?;
    let kind = g.string("general.architecture")?;
    if kind != KIND {
        return Err(err(format!("not a ferrum-gpu checkpoint (kind '{kind}')")));
    }
    let dim = |k: &str| get(&g, k).and_then(|v| narrow(v, format!("'{k}' is too large")));
    let cfg = Config {
        vocab: dim("vocab")?,
        ctx: dim("ctx")?,
        dim: dim("dim")?,
        heads: dim("heads")?,
        layers: dim("layers")?,
        hidden: dim("hidden")?,
    };
    let lens = param_lens(&cfg).ok_or_else(|| err("the checkpoint's model config is invalid"))?;
    if g.tensors.len() != tensors_for(lens.len())? {
        return Err(err("the tensor count does not match the model config"));
    }
    for (i, &n) in lens.iter().enumerate() {
        for kind in ["data", "m", "v"] {
            let name = format!("p{i}.{kind}");
            let t = g
                .tensor(&name)
                .ok_or_else(|| err(format!("missing tensor '{name}'")))?;
            if t.ggml_type != ferrum_gguf::GGML_F32 || t.dims != [n as u64] {
                return Err(err(format!(
                    "tensor '{name}' must be F32 with {n} elements"
                )));
            }
        }
    }
    let load = |kind: &str| -> Result<Gpt2, GgufError> {
        (0..lens.len())
            .map(|i| g.tensor_f32(&format!("p{i}.{kind}")))
            .collect::<Result<Vec<_>, _>>()
            .and_then(|params| Gpt2::from_params(cfg, params).map_err(err))
    };
    Ok(Checkpoint {
        snapshot: Snapshot {
            weights: load("data")?,
            m: load("m")?,
            v: load("v")?,
            steps: get(&g, "step")?,
        },
        epoch: get(&g, "epoch")?,
        step_in_epoch: get(&g, "step_in_epoch")?,
        shape: RunShape {
            context: get(&g, "shape.context")?,
            batch: get(&g, "shape.batch")?,
            stride: get(&g, "shape.stride")?,
            windows: get(&g, "shape.windows")?,
            seed: get(&g, "shape.seed")?,
        },
        corpus_id: g.string("ferrum.ckpt.corpus_id").ok().map(String::from),
    })
}
