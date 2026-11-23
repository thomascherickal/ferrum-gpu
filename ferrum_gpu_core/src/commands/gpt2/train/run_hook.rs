use super::*;

/// The `--metrics-csv` writer, with its header row.
fn metrics_csv(a: &Args) -> Result<Option<Box<dyn Write>>, String> {
    a.get("--metrics-csv")
        .map(|p| {
            let f = std::fs::File::create(p).map_err(|e| format!("{p}: {e}"))?;
            Ok::<_, String>(Box::new(with_header(f, p)?) as Box<dyn Write>)
        })
        .transpose()
}

/// Write the metrics header row to `w`, which is the file `p`.
pub(super) fn with_header<W: Write>(mut w: W, p: &str) -> Result<W, String> {
    w.write_all(b"step,epoch,loss,lr,grad_norm,tokens_per_sec,elapsed\n")
        .map_err(|e| format!("{p}: {e}"))?;
    Ok(w)
}

/// What the loop knows about the run when it starts.
pub(super) struct Shape<'a> {
    pub(super) tc: &'a TrainConfig,
    pub(super) steps_per_epoch: u64,
    pub(super) total: u64,
}

/// The step hook: progress lines, stop conditions, checkpoints and validation.
pub(super) fn run_hook<'a>(
    a: &Args,
    out: &'a mut dyn Write,
    target: &str,
    shape: Shape,
    val: Option<Validation>,
) -> Result<Run<'a>, String> {
    let Shape {
        tc,
        steps_per_epoch,
        total,
    } = shape;
    Ok(Run {
        out,
        verbose: a.has("--verbose"),
        csv: metrics_csv(a)?,
        started: Instant::now(),
        max_seconds: a
            .get("--max-seconds")
            .map(|_| a.num("--max-seconds", 0.0f64))
            .transpose()?,
        stop_file: Path::new(target).with_extension("stop"),
        checkpoint: a.get("--checkpoint").map(Into::into),
        sched: tc.cosine.then_some(Schedule {
            base_lr: tc.lr,
            warmup: tc.warmup,
            total,
        }),
        lr: tc.lr,
        tokens_per_step: tc.batch * tc.context,
        steps_per_epoch,
        total,
        val,
        best: None,
        bad_epochs: 0,
        last: None,
        reason: None,
        error: None,
    })
}
