//! Running a command on its own thread: its report is split into lines as it is written, its cancel
//! flag is installed for the thread, and its end (success, an error, or a panic) finishes the job.

use super::{Job, RunFn, State};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::thread;

/// A `Write` that turns what a command writes into the job's output lines. A line is complete at
/// `\n` (a trailing `\r` is dropped); what is left without a newline is flushed at the end.
pub struct LineSink {
    job: Arc<Job>,
    buf: Vec<u8>,
}

impl LineSink {
    /// A sink feeding `job`.
    pub fn new(job: Arc<Job>) -> LineSink {
        LineSink { job, buf: vec![] }
    }

    fn emit(&mut self) {
        let text = String::from_utf8_lossy(&self.buf)
            .trim_end_matches('\r')
            .to_string();
        self.buf.clear();
        self.job.push(false, text);
    }

    /// Emit the unfinished last line, if there is one.
    pub fn finish(&mut self) {
        if !self.buf.is_empty() {
            self.emit();
        }
    }
}

impl Write for LineSink {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        for &b in data {
            if b == b'\n' {
                self.emit();
            } else {
                self.buf.push(b);
            }
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The text of a panic payload.
fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

/// Start `run` for `job` on a new thread and return at once; the thread finishes the job.
pub fn spawn(run: Arc<RunFn>, job: Arc<Job>) -> Result<(), String> {
    let argv: Vec<String> = std::iter::once(job.cmd.clone())
        .chain(job.args.iter().cloned())
        .collect();
    started(
        thread::Builder::new()
            .name(format!("job-{}", job.id))
            .spawn(move || {
                let _cancel = crate::cancel::install(job.cancel.clone());
                let mut sink = LineSink::new(job.clone());
                let result = catch_unwind(AssertUnwindSafe(|| run(&argv, &mut sink)));
                sink.finish();
                match result {
                    Ok(Ok(())) => job.finish(State::Ended(0)),
                    Ok(Err(e)) => {
                        job.push(true, e);
                        job.finish(State::Ended(2));
                    }
                    Err(p) => {
                        job.push(true, format!("panicked: {}", panic_text(p.as_ref())));
                        job.finish(State::Ended(101));
                    }
                }
            }),
    )
}

/// `Ok` when the job's thread started; the reason it did not otherwise.
pub(in crate::jobs) fn started(r: std::io::Result<thread::JoinHandle<()>>) -> Result<(), String> {
    r.map(|_| ())
        .map_err(|e| format!("cannot start a thread for the job: {e}"))
}
