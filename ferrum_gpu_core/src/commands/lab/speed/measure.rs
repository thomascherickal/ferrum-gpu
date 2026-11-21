use super::*;

/// Measure one precision (`None` is f32).
pub fn measure(gpu: &Gpu, w: &Llama, quant: Option<QKind>, reps: usize) -> Result<Speed, String> {
    let c = w.cfg;
    let ids = ids(c.vocab);
    let mut m = GpuLlama::new(gpu, w, quant, None).map_err(text)?;
    let decode_ms = time_fallible(reps, || {
        m.reset();
        ids.iter()
            .try_for_each(|&t| m.feed(t).map(drop).map_err(text))
    })?;
    let prefill_ms = time_fallible(reps, || m.forward_all(&ids).map(drop).map_err(text))?;
    Ok(Speed {
        precision: quant.map_or("f32", QKind::label).into(),
        model_mib: plan(&c, quant).total() as f64 / 1048576.0,
        decode_tps: tps(TOKENS, decode_ms),
        prefill_tps: tps(TOKENS, prefill_ms),
    })
}

/// The CPU reference's prefill speed on the same model, for scale.
pub fn cpu_prefill(w: &Llama, reps: usize) -> Result<f64, String> {
    let ids = ids(w.cfg.vocab);
    let ms = time_fallible(reps, || w.forward_logits(&ids).map(drop))?;
    Ok(tps(TOKENS, ms))
}

/// Every precision.
pub fn run(gpu: &Gpu, size: Size, reps: usize) -> Result<(Vec<Speed>, f64), String> {
    let w = Llama::random(size.config(), 5, false, false);
    let mut speeds = Vec::new();
    for q in [
        None,
        Some(QKind::F16),
        Some(QKind::Q8_0),
        Some(QKind::Q4_0),
        Some(QKind::Fp4),
        Some(QKind::Fp8E4M3),
        Some(QKind::Fp8E5M2),
        Some(QKind::Q2K),
    ] {
        speeds.push(measure(gpu, &w, q, reps)?);
    }
    Ok((speeds, cpu_prefill(&w, reps)?))
}
