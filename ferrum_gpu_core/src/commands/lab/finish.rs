use super::*;

/// Print a finished report (or the reason there is none) and turn a failed parity check into an error.
pub(super) fn finish(
    report: Result<Report, String>,
    json: bool,
    out: &mut dyn Write,
) -> Result<(), String> {
    let report = report?;
    let shown = if json {
        report.json().to_string()
    } else {
        report.text()
    };
    writeln!(out, "{shown}").map_err(text)?;
    if report.passed() {
        Ok(())
    } else {
        Err("lab: a parity check failed".into())
    }
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
    );
    finish(report, a.has("--json"), out)
}
