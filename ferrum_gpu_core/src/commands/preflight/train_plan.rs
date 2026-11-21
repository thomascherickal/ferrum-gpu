use super::*;

/// Buffers a training run needs: weights, gradients, both Adam moments and the activations of a
/// `(batch, context)` step.
pub fn train_plan(m: &Llama, batch: usize, context: usize) -> BufferPlan {
    let mut p = static_plan(m);
    for (n, b) in activation_plan(&m.cfg, batch, context).rows() {
        p.add(n, *b);
    }
    p
}

/// Buffers a decoder needs: weights, tables, KV cache and scratch.
pub fn decode_buffers(m: &Llama) -> BufferPlan {
    decode_plan(&m.cfg, None)
}

/// The same, with every weight larger than the chosen adapter's binding limit listed as the blocks
/// the engine stores it in (a CPU backend, or an unknown adapter, has no limit).
pub fn decode_buffers_for(m: &Llama, app: &crate::session::Session) -> BufferPlan {
    let name = app.backend.name();
    let limit = app
        .adapters
        .iter()
        .find(|a| a.name == name && app.backend.kind() == "gpu")
        .map_or(u64::MAX, |a| a.max_binding);
    ferrum_wgpu::llama::plan_limited(&m.cfg, app.weights, limit)
}

/// Buffers a GPT-2 decoder needs: weights, KV cache and scratch.
pub fn gpt2_decode_buffers(m: &ferrum_ref::gpt2::Gpt2) -> BufferPlan {
    ferrum_wgpu::gpt2::plan(&m.cfg)
}

/// The budget `--gpu-mem-mb` sets, or `None` when it is not given. What the machine reports instead
/// (free VRAM, available RAM) is read and interpreted in `sysmem` and `ferrum_wgpu::memory`.
pub fn override_budget(mem_mb: Option<u64>) -> Option<Budget> {
    ferrum_wgpu::memory::resolve_memory(mem_mb, false, &ferrum_wgpu::memory::MemoryFacts::default())
}

/// Mebibytes, one decimal.
pub(super) fn mib(b: u64) -> String {
    format!("{:.1} MiB", b as f64 / 1048576.0)
}

/// The line to print on success.
pub fn report(plan: &BufferPlan, budget: Option<Budget>) -> String {
    let have = budget.map_or("unknown".to_string(), |b| mib(b.bytes));
    format!(
        "preflight: GPU buffers {} in {} allocations (budget {have})",
        mib(plan.total()),
        plan.rows().len()
    )
}

/// The same line with the budget's source spelled out (`describe`), for `info` and the GUI.
pub fn explain(budget: Option<Budget>, facts: &ferrum_wgpu::memory::MemoryFacts) -> String {
    budget.map_or(
        "memory: unknown (only the largest-buffer limit is checked; pass --gpu-mem-mb to set a budget)"
            .to_string(),
        |b| format!("memory budget: {}", ferrum_wgpu::memory::describe(&b, facts)),
    )
}

/// Refuse if `plan` does not fit `adapter` and `budget`. The message says what to do instead.
pub fn check(
    plan: &BufferPlan,
    adapter: &AdapterSummary,
    budget: Option<Budget>,
) -> Result<(), String> {
    plan.check(adapter.max_binding, budget).map_err(|e| {
        format!(
            "refusing to start on {}: {e} (needs {} in total); use a smaller preset, batch or \
             context, raise --gpu-mem-mb, or run with --device cpu",
            adapter.name,
            mib(plan.total())
        )
    })
}
