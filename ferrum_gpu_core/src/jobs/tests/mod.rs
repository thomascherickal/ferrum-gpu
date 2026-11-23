//! Tests of the jobs module. Commands are replaced by fake runners (closures), so the tests need no
//! GPU, no shell and no binary; one test runs the real `info` command in-process.

mod allow;
mod progress;
mod runner;

pub(crate) mod support {
    use crate::jobs::{Job, Jobs, RunFn};
    use std::io::Write;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// A job table whose commands are `f(argv, out)`.
    pub(super) fn jobs_with(
        f: impl Fn(&[String], &mut dyn Write) -> Result<(), String> + Send + Sync + 'static,
    ) -> Jobs {
        let run: Arc<RunFn> = Arc::new(f);
        Jobs::with_runner(run)
    }

    pub(crate) fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// Wait until the job has finished (panic after `secs`).
    pub(crate) fn wait(job: &Arc<Job>, secs: u64) {
        let t0 = Instant::now();
        while !job.is_finished() {
            assert!(
                t0.elapsed() < Duration::from_secs(secs),
                "job {} ({}) did not finish: {:?}",
                job.id,
                job.cmd,
                job.poll(0).lines
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Wait until some output line contains `needle`.
    pub(super) fn wait_for_line(job: &Arc<Job>, needle: &str) {
        let t0 = Instant::now();
        while !job
            .poll(0)
            .lines
            .iter()
            .any(|(_, l)| l.text.contains(needle))
        {
            assert!(
                t0.elapsed() < Duration::from_secs(20),
                "no line with {needle:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    pub(crate) fn texts(job: &Arc<Job>) -> Vec<String> {
        job.poll(0).lines.into_iter().map(|(_, l)| l.text).collect()
    }

    /// Block until this thread's cancel flag is raised (the way a training loop waits for a stop).
    pub(super) fn block_until_cancelled() {
        let t0 = Instant::now();
        while !crate::cancel::requested() {
            assert!(t0.elapsed() < Duration::from_secs(20), "never cancelled");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
