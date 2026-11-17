//! Tests of the jobs module. `ferrum-gpu` is replaced by a stub shell script (unix only), so the
//! tests need no GPU and no real binary.

mod allow;
mod cli;
mod progress;
#[cfg(unix)]
mod routes;
#[cfg(unix)]
mod runner;

#[cfg(unix)]
pub(crate) mod support {
    use crate::jobs::{Job, Jobs};
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// The stub: behaviour is chosen by the subcommand the Studio runs.
    const STUB: &str = r#"#!/bin/sh
cmd="$1"; shift
all="$*"
out=""
while [ $# -gt 0 ]; do
  case "$1" in --out|--model) out="$2";; esac
  shift 2>/dev/null || break
done
case "$cmd" in
  info) echo "args: $all"; echo "device: stub";;
  train)
    echo "step 10/100 loss 3.0000 grad_norm 0.500 tokens/s 1000 elapsed 1.0s eta 9s"
    echo "epoch 1 val loss 2.5000 (best)"
    echo "step 20/100 loss 2.0000 grad_norm 0.400 tokens/s 1100 elapsed 2.0s eta 8s"
    stopfile="${out%.*}.stop"
    echo "waiting for $stopfile"
    n=0
    while [ ! -e "$stopfile" ] && [ $n -lt 600 ]; do sleep 0.05; n=$((n+1)); done
    echo "stop file seen";;
  eval) echo "starting"; echo "something went wrong" >&2; exit 3;;
  tokenize) i=0; while [ $i -lt 25000 ]; do echo "line $i"; i=$((i+1)); done;;
  prepare) echo "sleeping"; sleep 3;;
  run) echo "ignoring the stop file"; trap '' TERM; sleep 3;;
  generate) printf 'no newline at the end';;
  export-gguf) printf 'crlf line\r\n';;
  *) echo "unexpected $cmd" >&2; exit 9;;
esac
"#;

    static N: AtomicU32 = AtomicU32::new(0);

    /// A fresh temp directory.
    pub fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ferrum_jobs_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The stub CLI, executable, in a fresh directory; returns `(dir, path)`.
    pub fn stub() -> (PathBuf, PathBuf) {
        let dir = tmp();
        let p = dir.join("ferrum-gpu");
        std::fs::write(&p, STUB).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        (dir, p)
    }

    /// A job table that runs the stub; the grace period is short so a kill test is quick.
    pub fn jobs() -> (Jobs, PathBuf) {
        let (dir, p) = stub();
        (Jobs::with_grace(Some(p), Duration::from_millis(400)), dir)
    }

    pub fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// Wait until the job has finished (panic after `secs`).
    pub fn wait(job: &Arc<Job>, secs: u64) {
        let t0 = Instant::now();
        while !job.is_finished() {
            assert!(
                t0.elapsed() < Duration::from_secs(secs),
                "job {} ({}) did not finish: {:?}",
                job.id,
                job.cmd,
                job.poll(0).lines
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Wait until some output line contains `needle`.
    pub fn wait_for_line(job: &Arc<Job>, needle: &str) {
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
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn texts(job: &Arc<Job>) -> Vec<String> {
        job.poll(0).lines.into_iter().map(|(_, l)| l.text).collect()
    }

    pub fn exists(p: &Path) -> bool {
        p.exists()
    }
}
