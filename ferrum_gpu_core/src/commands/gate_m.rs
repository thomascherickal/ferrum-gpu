use super::*;

/// Pre-flight: when the backend is a GPU, refuse a run whose buffers do not fit (see
/// [`preflight`]) before any of it is uploaded, else print the estimate. A CPU backend, or an
/// adapter missing from the list (limits unknown), passes.
pub fn gate(
    out: &mut dyn Write,
    app: &Session,
    plan: &ferrum_wgpu::budget::BufferPlan,
    mem_mb: Option<u64>,
) -> Result<(), String> {
    gate_with(&sysmem::System, out, app, plan, mem_mb)
}

/// [`gate`] reading the machine's memory through `readers` (tests give fixtures).
pub fn gate_with(
    readers: &dyn sysmem::Readers,
    out: &mut dyn Write,
    app: &Session,
    plan: &ferrum_wgpu::budget::BufferPlan,
    mem_mb: Option<u64>,
) -> Result<(), String> {
    let name = app.backend.name();
    let adapter = app.adapters.iter().find(|a| a.name == name);
    let Some(adapter) = adapter.filter(|_| app.backend.kind() == "gpu") else {
        return Ok(());
    };
    let (budget, _) = sysmem::budget_for(mem_mb, adapter, readers);
    preflight::check(plan, adapter, budget)?;
    say(out, preflight::report(plan, budget))
}

pub(super) fn mem_mb(a: &Args) -> Result<Option<u64>, String> {
    a.get("--gpu-mem-mb")
        .map(|_| a.num("--gpu-mem-mb", 0u64))
        .transpose()
}

pub(super) fn say(out: &mut dyn Write, s: String) -> Result<(), String> {
    writeln!(out, "{s}").map_err(|e| e.to_string())
}

pub(super) fn device_line(out: &mut dyn Write, app: &Session) -> Result<(), String> {
    say(
        out,
        format!("device: {} ({})", app.backend.name(), app.backend.kind()),
    )?;
    app.fallback_note
        .as_ref()
        .map_or(Ok(()), |n| say(out, format!("note: {n}")))
}

/// Read a GGUF model file.
pub fn load(path: &str) -> Result<Llama, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let g = Gguf::parse(bytes).map_err(|e| format!("{path}: {e}"))?;
    from_gguf(&g).map_err(|e| format!("{path}: {e}"))
}

pub(super) fn shape(m: &Llama) -> String {
    let c = &m.cfg;
    format!(
        "model: vocab {} ctx {} dim {} heads {}/{} layers {} hidden {}",
        c.vocab, c.ctx, c.dim, c.heads, c.kv_heads, c.layers, c.hidden
    )
}

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

pub(super) fn info(a: &Args, out: &mut dyn Write) -> Result<(), String> {
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
