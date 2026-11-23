use super::*;

/// Build the app (device) from the shared device flags.
pub(crate) fn app(a: &Args, work: Workload) -> Result<Session, String> {
    app_requested(
        a,
        work,
        DeviceChoice::parse(a.get("--device").unwrap_or("auto"))?,
    )
}

/// [`app`] with the device choice decided by the caller instead of read from `--device`.
fn app_requested(a: &Args, work: Workload, requested: DeviceChoice) -> Result<Session, String> {
    let threads = a.num(
        "--threads",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )?;
    let d = decide(requested, work, &Gpu::list(), a.get("--gpu-adapter"))?;
    let weights = gpu_weights(a.get("--gpu-weights"))?;
    let mut app = Session::new_with(d.choice, d.adapter.as_deref(), threads.max(1), weights)?;
    if d.note.is_some() {
        app.fallback_note = d.note;
    }
    Ok(app)
}
