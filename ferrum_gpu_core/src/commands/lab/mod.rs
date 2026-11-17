//! `ferrum-gpu lab`: one report about a machine's GPU. It checks that the GPU agrees with the CPU
//! reference (f32, Q8_0, Q4_0, and weights stored in blocks), measures decode and prefill speed per
//! precision, and prints the adapter, driver, features and memory budget. `--json` writes the same
//! as a file an owner of hardware we do not have can send back. The Studio's GPU lab tab, the
//! kernel benchmarks and the verification kit are all this command.

mod parity;
pub mod report;
mod speed;

use crate::commands::flags::{switch, val, Args, Flag};
use crate::commands::{preflight, sysmem};
use ferrum_wgpu::Gpu;
use report::{AdapterFacts, Report};
use std::io::Write;

/// The flags `lab` accepts.
pub const FLAGS: [Flag; 5] = [
    val("--gpu-adapter"),
    val("--reps"),
    val("--size"),
    switch("--json"),
    switch("--parity-only"),
];

/// Most timed repetitions per measurement.
const MAX_REPS: usize = 200;

fn facts(gpu: &Gpu, readers: &dyn sysmem::Readers) -> AdapterFacts {
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

/// The `lab` command.
pub fn run(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let reps: usize = a.num("--reps", 5)?;
    if reps == 0 || reps > MAX_REPS {
        return Err(format!("--reps must be between 1 and {MAX_REPS}"));
    }
    let size = speed::Size::parse(a.get("--size").unwrap_or("small"))?;
    let gpu =
        Gpu::open(a.get("--gpu-adapter")).map_err(|e| format!("lab needs a GPU adapter: {e}"))?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let report = build(
        &gpu,
        (size, reps, !a.has("--parity-only")),
        &sysmem::System,
        now,
    )?;
    let text = if a.has("--json") {
        report.json().to_string()
    } else {
        report.text()
    };
    writeln!(out, "{text}").map_err(|e| e.to_string())?;
    if report.passed() {
        Ok(())
    } else {
        Err("lab: a parity check failed".into())
    }
}

#[cfg(test)]
mod tests;
