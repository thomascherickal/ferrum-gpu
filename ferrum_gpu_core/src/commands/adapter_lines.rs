use super::*;

/// What `info` prints about one adapter: its identity, its features and largest buffer, and the memory
/// budget with where it came from.
pub(super) fn adapter_lines(
    ad: &ferrum_wgpu::select::AdapterSummary,
    mem: Option<u64>,
) -> [String; 3] {
    let feats = ad.features.words();
    let (budget, facts) = sysmem::budget_for(mem, ad, &sysmem::System);
    [
        format!("adapter: {} ({:?}, {})", ad.name, ad.kind, ad.backend),
        format!(
            "  features: {}; largest buffer {} MiB",
            if feats.is_empty() {
                "none reported".to_string()
            } else {
                feats.join(", ")
            },
            ad.max_binding >> 20
        ),
        format!("  {}", preflight::explain(budget, &facts)),
    ]
}

pub(crate) fn info(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let app = app(a, Workload::Decode)?;
    device_line(out, &app)?;
    let mem = mem_mb(a)?;
    for ad in &app.adapters {
        for line in adapter_lines(ad, mem) {
            say(out, line)?;
        }
    }
    if let Some(p) = a.get("--model") {
        say(
            out,
            if gpt2::is_gpt2(p) {
                gpt2::shape_of(p)?
            } else {
                shape(&load(p)?)
            },
        )?;
    }
    Ok(())
}

/// `--gpu-weights`: how a GPU decoder stores weights (`f32` is the default and means no conversion).
pub fn gpu_weights(v: Option<&str>) -> Result<Option<ferrum_wgpu::ops_quant::QKind>, String> {
    use ferrum_wgpu::ops_quant::QKind;
    match v.unwrap_or("f32") {
        "f32" => Ok(None),
        "f16" => Ok(Some(QKind::F16)),
        "q8_0" => Ok(Some(QKind::Q8_0)),
        "q4_0" => Ok(Some(QKind::Q4_0)),
        "fp4" | "mxfp4" => Ok(Some(QKind::Fp4)),
        "fp8" | "fp8e4m3" => Ok(Some(QKind::Fp8E4M3)),
        "fp8e5m2" => Ok(Some(QKind::Fp8E5M2)),
        "q2" | "q2_k" => Ok(Some(QKind::Q2K)),
        o => Err(format!(
            "--gpu-weights: unknown '{o}' (f32, f16, q8_0, q4_0, fp4, fp8 (E4M3), fp8e5m2, q2_k)"
        )),
    }
}
