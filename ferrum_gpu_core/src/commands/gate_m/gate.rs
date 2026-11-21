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

pub(crate) fn mem_mb(a: &Args) -> Result<Option<u64>, String> {
    a.get("--gpu-mem-mb")
        .map(|_| a.num("--gpu-mem-mb", 0u64))
        .transpose()
}

pub(crate) fn say(out: &mut dyn Write, s: String) -> Result<(), String> {
    writeln!(out, "{s}").map_err(|e| e.to_string())
}

pub(crate) fn device_line(out: &mut dyn Write, app: &Session) -> Result<(), String> {
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

pub(crate) fn shape(m: &Llama) -> String {
    let c = &m.cfg;
    format!(
        "model: vocab {} ctx {} dim {} heads {}/{} layers {} hidden {}",
        c.vocab, c.ctx, c.dim, c.heads, c.kv_heads, c.layers, c.hidden
    )
}
