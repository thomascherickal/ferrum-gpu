use super::*;

/// The `model:` and `corpus:` lines, printed before the memory gate.
pub(super) fn announce(
    out: &mut dyn Write,
    cfg: &Config,
    train_len: usize,
    tc: &TrainConfig,
    total: u64,
) -> Result<(), String> {
    say(
        out,
        format!(
            "model: vocab {} ctx {} dim {} heads {} layers {} hidden {}",
            cfg.vocab, cfg.ctx, cfg.dim, cfg.heads, cfg.layers, cfg.hidden
        ),
    )?;
    say(
        out,
        format!(
            "corpus: {} tokens, {} windows, {total} steps of {}",
            train_len,
            window_starts(train_len, tc.context, tc.stride).len(),
            tc.batch
        ),
    )
}

/// Refuse a run whose static weights plus activations do not fit the device (or `--mem`).
pub(super) fn gate_memory(
    a: &Args,
    out: &mut dyn Write,
    app: &Session,
    cfg: &Config,
    tc: &TrainConfig,
) -> Result<(), String> {
    let mut plan = ferrum_wgpu::train::static_plan(cfg);
    for (n, b) in ferrum_wgpu::train::activation_plan(cfg, tc.batch, tc.context).rows() {
        plan.add(n, *b);
    }
    gate(out, app, &plan, mem_mb(a)?)
}
