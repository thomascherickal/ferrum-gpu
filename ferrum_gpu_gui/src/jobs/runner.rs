//! Starting the process and following it: two reader threads (stdout, stderr) feed the job's line
//! buffer, and one waiter thread ends the process when asked (a stop file that was not honoured in
//! time, or a forced stop). The process is ended with `Child::kill` only, never by signalling a
//! process group (rule 16).

use super::{Job, State};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// How often the waiter checks on the process.
const TICK: Duration = Duration::from_millis(25);
/// How long the waiter lets the output readers drain after the process has ended.
const READER_PATIENCE: Duration = Duration::from_secs(1);

fn read_lines(job: Arc<Job>, src: impl Read + Send + 'static, err: bool) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(src).split(b'\n').map_while(Result::ok) {
            let text = String::from_utf8_lossy(&line)
                .trim_end_matches('\r')
                .to_string();
            job.push(err, text);
        }
    })
}

/// Start `exe <cmd> <args...>` for `job` and return at once; the threads finish the job.
pub fn spawn(exe: &Path, job: Arc<Job>) -> Result<(), String> {
    // a `.stop` file left over from an earlier run would end this one at its first step
    if let Some(f) = &job.stop_file {
        if f.exists() && std::fs::remove_file(f).is_ok() {
            job.push(false, format!("(removed a stale {})", f.display()));
        }
    }
    let mut child = Command::new(exe)
        .arg(&job.cmd)
        .args(&job.args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
    let readers = [
        child
            .stdout
            .take()
            .map(|s| read_lines(job.clone(), s, false)),
        child
            .stderr
            .take()
            .map(|s| read_lines(job.clone(), s, true)),
    ];
    thread::spawn(move || {
        let status = loop {
            match child.try_wait() {
                Ok(Some(s)) => break Ok(s),
                Ok(None) => {}
                Err(e) => break Err(e.to_string()),
            }
            let due = job
                .deadline
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some_and(|d| Instant::now() >= d);
            if job.kill_now.load(Ordering::SeqCst) || due {
                job.push(false, "(the Studio ended the process)".into());
                let _ = child.kill();
                job.kill_now.store(false, Ordering::SeqCst);
            }
            thread::sleep(TICK);
        };
        // The readers end when every holder of the pipe has closed it. A grandchild of a killed
        // process can keep it open, so wait a moment for the last lines and then let go.
        let t0 = Instant::now();
        for r in readers.into_iter().flatten() {
            while !r.is_finished() && t0.elapsed() < READER_PATIENCE {
                thread::sleep(Duration::from_millis(5));
            }
            if r.is_finished() {
                let _ = r.join();
            }
        }
        if let Some(f) = &job.stop_file {
            let _ = std::fs::remove_file(f);
        }
        // a process ended by a signal has no exit code; report it as -1
        job.finish(match status {
            Ok(s) => State::Ended(s.code().unwrap_or(-1)),
            Err(e) => State::Failed(e),
        });
    });
    Ok(())
}
