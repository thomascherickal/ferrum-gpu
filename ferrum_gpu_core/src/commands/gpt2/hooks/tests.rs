use super::*;

/// A writer that is already closed.
struct Closed;

impl Write for Closed {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("closed"))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn run<'a>(out: &'a mut dyn Write, csv: Option<Box<dyn Write>>, verbose: bool) -> Run<'a> {
    Run {
        out,
        verbose,
        csv,
        started: Instant::now(),
        max_seconds: None,
        stop_file: PathBuf::from("/nonexistent-ferrum/never.stop"),
        checkpoint: None,
        sched: None,
        lr: 0.01,
        tokens_per_step: 4,
        steps_per_epoch: 2,
        total: 1,
        val: None,
        best: None,
        bad_epochs: 0,
        last: None,
        reason: None,
        error: None,
    }
}

const STATS: StepStats = StepStats {
    loss: 1.0,
    grad_norm: 1.0,
};

#[test]
fn a_metrics_row_that_cannot_be_written_stops_the_run_and_names_the_flag() {
    let mut sink = Vec::new();
    let mut r = run(&mut sink, Some(Box::new(Closed)), false);
    assert!(!r.on_step(1, &STATS));
    assert!(r.error.unwrap().starts_with("--metrics-csv: closed"));
}

#[test]
fn a_progress_line_that_cannot_be_written_is_remembered_for_the_end_of_the_run() {
    let mut closed = Closed;
    let mut r = run(&mut closed, None, true);
    // the last step prints a progress line; the failure does not stop training
    assert!(r.on_step(1, &STATS));
    assert_eq!(r.error.as_deref(), Some("closed"));
}

#[test]
fn only_the_first_write_failure_is_kept() {
    let mut closed = Closed;
    let mut r = run(&mut closed, None, true);
    r.note("a".into());
    r.note("b".into());
    assert_eq!(r.error.as_deref(), Some("closed"));
}

#[test]
fn a_cancelled_request_stops_the_run_after_the_step() {
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let _scope = crate::cancel::install(flag);
    let mut sink = Vec::new();
    let mut r = run(&mut sink, None, false);
    assert!(!r.on_step(1, &STATS));
    assert_eq!(r.reason.as_deref(), Some("cancelled; stopped after step 1"));
}
