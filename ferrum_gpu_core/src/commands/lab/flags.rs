use super::*;

/// The flags `lab` accepts.
pub const FLAGS: [Flag; 5] = [
    val("--gpu-adapter"),
    val("--reps"),
    val("--size"),
    switch("--json"),
    switch("--parity-only"),
];

/// Most timed repetitions per measurement.
pub(super) const MAX_REPS: usize = 200;

pub(super) fn facts(gpu: &Gpu, readers: &dyn sysmem::Readers) -> AdapterFacts {
    let info = gpu.info();
    let driver = ferrum_wgpu::device::drivers()
        .into_iter()
        .find(|d| d.0 == info.name)
        .map(|d| d.1)
        .unwrap_or_default();
    let (budget, memory) = sysmem::budget_for(None, info, readers);
    AdapterFacts {
        name: info.name.clone(),
        backend: info.backend.clone(),
        kind: format!("{:?}", info.kind),
        driver,
        max_binding_mib: info.max_binding >> 20,
        features: info
            .features
            .words()
            .into_iter()
            .map(String::from)
            .collect(),
        budget: preflight::explain(budget, &memory),
    }
}

/// Build the report on an opened adapter. `speed` false skips the timings (parity only).
pub fn build(
    gpu: &Gpu,
    (size, reps, speed): (speed::Size, usize, bool),
    readers: &dyn sysmem::Readers,
    now_secs: u64,
) -> Result<Report, String> {
    let mut r = Report {
        when: report::date_of(now_secs),
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        wgpu: report::locked_version(include_str!("../../../../Cargo.lock"), "wgpu"),
        adapter: Some(facts(gpu, readers)),
        reps,
        checks: parity::run(gpu)?,
        ..Report::default()
    };
    if speed {
        let (speeds, cpu) = speed::run(gpu, size, reps)?;
        r.model = size.describe();
        r.speeds = speeds;
        r.cpu_prefill_tps = Some(cpu);
        r.notes.push(
            "the model is small, so these speeds are dominated by launch and dispatch cost, not memory bandwidth; they are not a large-model figure".into(),
        );
    }
    Ok(r)
}

/// The message of a failure, for `map_err` (one function instead of a closure at every call).
pub(crate) fn text(e: impl std::fmt::Display) -> String {
    e.to_string()
}
