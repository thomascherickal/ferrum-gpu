use super::*;

/// The small model the checks use (every width a whole number of quantisation blocks except the
/// head width, as in the engine's own tests).
pub(super) fn config() -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        layers: 2,
        hidden: 64,
        rope_dim: 8,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

/// A model whose matrices are whole 256-element blocks, for Q2_K (the small one has none).
pub(super) fn config_k() -> Config {
    Config {
        dim: 256,
        head_dim: 64,
        layers: 1,
        hidden: 512,
        rope_dim: 64,
        ..config()
    }
}

pub(super) const IDS: [usize; 10] = [3, 16, 0, 9, 9, 4, 11, 2, 7, 5];
/// Larger than the biggest buffer that cannot be split, smaller than every weight: it makes the
/// model store its weights and embedding as blocks.
pub(super) const BLOCK_LIMIT: u64 = 1000;

pub(super) fn check(name: &str, error: f32) -> Check {
    Check {
        name: name.into(),
        error,
        tolerance: FORWARD,
    }
}

/// Prefill of `w` stored as `kind` on the GPU against the CPU reference over the dequantised model.
pub(super) fn quant_check(gpu: &Gpu, w: &Llama, kind: QKind) -> Result<Check, String> {
    let oracle = dequantised(w, kind).map_err(text)?;
    let want = oracle.forward_logits(&IDS).map_err(text)?;
    let m = GpuLlama::new(gpu, w, Some(kind), None).map_err(text)?;
    Ok(check(
        &format!(
            "prefill, {} (against the dequantised reference)",
            kind.label()
        ),
        max_rel_err(&m.forward_all(&IDS).map_err(text)?, &want),
    ))
}
