use super::*;

pub(super) fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

pub(super) fn opt_uint(g: &Gguf, key: &str) -> Result<Option<usize>, GgufError> {
    g.get(key).map(|_| g.uint(key)).transpose()
}

pub(super) fn opt_float(g: &Gguf, key: &str) -> Result<Option<f32>, GgufError> {
    g.get(key).map(|_| g.float(key)).transpose()
}

/// Read a tensor that must have exactly the GGML dims `dims`, as f32.
pub(super) fn load(g: &Gguf, name: &str, dims: &[usize]) -> Result<Vec<f32>, GgufError> {
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
pub(super) fn load_matrix(
    g: &Gguf,
    name: &str,
    inp: usize,
    out: usize,
) -> Result<Vec<f32>, GgufError> {
    Ok(transpose(&load(g, name, &[inp, out])?, out, inp))
}

/// An optional bias vector (empty when the file has none).
pub(super) fn load_bias(g: &Gguf, name: &str, n: usize) -> Result<Vec<f32>, GgufError> {
    if g.tensor(name).is_some() {
        load(g, name, &[n])
    } else {
        Ok(Vec::new())
    }
}

/// RoPE scaling as llama.cpp reads it. `yarn`, `longrope` and a non-1 `attn_factor` are refused:
/// running them unscaled would be silently wrong.
pub(super) fn rope_scaling(
    g: &Gguf,
    arch: &str,
    rope_dim: usize,
) -> Result<RopeScaling, GgufError> {
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
