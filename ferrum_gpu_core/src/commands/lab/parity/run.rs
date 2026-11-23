use super::*;

/// Run every check; the first device error aborts the report.
pub(crate) fn run(gpu: &Gpu) -> Result<Vec<Check>, String> {
    let w = Llama::random(config(), 11, true, false);
    let want = w.forward_logits(&IDS).map_err(text)?;
    let v = config().vocab;
    let mut out = Vec::new();

    let m = GpuLlama::new(gpu, &w, None, None).map_err(text)?;
    out.push(check(
        "prefill, f32",
        max_rel_err(&m.forward_all(&IDS).map_err(text)?, &want),
    ));

    let mut m = GpuLlama::new(gpu, &w, None, None).map_err(text)?;
    let mut worst = 0f32;
    for (i, &t) in IDS.iter().enumerate() {
        let row = m.feed(t).map_err(text)?;
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

    let whole = GpuLlama::new(gpu, &w, None, None).map_err(text)?;
    let cut = GpuLlama::with_binding_limit(gpu, &w, None, None, BLOCK_LIMIT).map_err(text)?;
    out.push(check(
        &format!("weights stored in blocks of at most {BLOCK_LIMIT} bytes, against stored whole"),
        max_rel_err(
            &cut.forward_all(&IDS).map_err(text)?,
            &whole.forward_all(&IDS).map_err(text)?,
        ),
    ));
    Ok(out)
}
