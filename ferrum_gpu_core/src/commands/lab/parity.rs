//! Parity checks: the GPU against the CPU reference on a small random model. Needs a device, so
//! the arithmetic (what counts as a pass) lives in `report`; this file only runs the models.

use super::report::Check;
use ferrum_ref::llama::{Config, Llama, RopeType};
use ferrum_wgpu::llama::{dequantised, GpuLlama};
use ferrum_wgpu::ops_quant::QKind;
use ferrum_wgpu::tolerance::{max_rel_err, FORWARD};
use ferrum_wgpu::Gpu;

/// The small model the checks use (every width a whole number of quantisation blocks except the
/// head width, as in the engine's own tests).
fn config() -> Config {
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
fn config_k() -> Config {
    Config {
        dim: 256,
        head_dim: 64,
        layers: 1,
        hidden: 512,
        rope_dim: 64,
        ..config()
    }
}

const IDS: [usize; 10] = [3, 16, 0, 9, 9, 4, 11, 2, 7, 5];
/// Larger than the biggest buffer that cannot be split, smaller than every weight: it makes the
/// model store its weights and embedding as blocks.
const BLOCK_LIMIT: u64 = 1000;

fn check(name: &str, error: f32) -> Check {
    Check {
        name: name.into(),
        error,
        tolerance: FORWARD,
    }
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Prefill of `w` stored as `kind` on the GPU against the CPU reference over the dequantised model.
fn quant_check(gpu: &Gpu, w: &Llama, kind: QKind) -> Result<Check, String> {
    let oracle = dequantised(w, kind).map_err(err)?;
    let want = oracle.forward_logits(&IDS).map_err(err)?;
    let m = GpuLlama::new(gpu, w, Some(kind), None).map_err(err)?;
    Ok(check(
        &format!(
            "prefill, {} (against the dequantised reference)",
            kind.label()
        ),
        max_rel_err(&m.forward_all(&IDS).map_err(err)?, &want),
    ))
}

/// Run every check; the first device error aborts the report.
pub fn run(gpu: &Gpu) -> Result<Vec<Check>, String> {
    let w = Llama::random(config(), 11, true, false);
    let want = w.forward_logits(&IDS).map_err(err)?;
    let v = config().vocab;
    let mut out = Vec::new();

    let m = GpuLlama::new(gpu, &w, None, None).map_err(err)?;
    out.push(check(
        "prefill, f32",
        max_rel_err(&m.forward_all(&IDS).map_err(err)?, &want),
    ));

    let mut m = GpuLlama::new(gpu, &w, None, None).map_err(err)?;
    let mut worst = 0f32;
    for (i, &t) in IDS.iter().enumerate() {
        let row = m.feed(t).map_err(err)?;
        worst = worst.max(max_rel_err(&row, &want[i * v..(i + 1) * v]));
    }
    out.push(check("decode with the KV cache, f32", worst));

    for kind in [
        QKind::F16,
        QKind::Q8_0,
        QKind::Q4_0,
        QKind::Fp4,
        QKind::Fp8E4M3,
        QKind::Fp8E5M2,
    ] {
        out.push(quant_check(gpu, &w, kind)?);
    }
    let wk = Llama::random(config_k(), 12, true, false);
    out.push(quant_check(gpu, &wk, QKind::Q2K)?);

    let whole = GpuLlama::new(gpu, &w, None, None).map_err(err)?;
    let cut = GpuLlama::with_binding_limit(gpu, &w, None, None, BLOCK_LIMIT).map_err(err)?;
    out.push(check(
        &format!("weights stored in blocks of at most {BLOCK_LIMIT} bytes, against stored whole"),
        max_rel_err(
            &cut.forward_all(&IDS).map_err(err)?,
            &whole.forward_all(&IDS).map_err(err)?,
        ),
    ));
    Ok(out)
}

#[cfg(test)]
mod tests;
