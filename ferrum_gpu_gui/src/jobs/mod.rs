//! Jobs: `ferrum-gpu` subcommands run as child processes, with their output streamed to the page.
//!
//! `ferrum_gpu_cli` depends on this crate, so the GUI cannot call the CLI's code; it runs the CLI
//! instead. That keeps one implementation of every command, and a GPU fault ends the job rather than
//! the GUI. What can be run is fixed in [`allow`]; where the binary is found is in [`cli`]; the
//! progress lines are read in [`progress`]; the process handling is in [`runner`]; the JSON the page
//! reads is in [`view`]; the HTTP routes are in [`routes`].

pub mod allow;
pub mod cli;
pub mod progress;
pub mod routes;
pub mod runner;
pub mod view;

#[cfg(test)]
mod tests;

use progress::Event;
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Most output lines kept per job (older ones are dropped, and `since` stays valid).
pub const MAX_LINES: usize = 20_000;
/// Most jobs running at once.
pub const MAX_RUNNING: usize = 4;
/// Most jobs remembered; the oldest finished ones are forgotten first.
pub const MAX_KEPT: usize = 50;
/// How long a stop request waits for a clean exit before the process is killed.
pub const GRACE: Duration = Duration::from_secs(20);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One line of a job's output.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// Came from standard error.
    pub err: bool,
    /// The text, without the line ending.
    pub text: String,
    /// The progress event it carries, if it is a progress line.
    pub ev: Option<Event>,
}

/// Where a job stands.
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// The process is running.
    Running,
    /// The process ended; its exit status.
    Ended(i32),
    /// The process could not be started or waited for.
    Failed(String),
}

pub(crate) struct Inner {
    pub lines: VecDeque<Line>,
    /// Lines dropped from the front so far.
    pub base: usize,
    pub state: State,
    pub last_step: Option<Event>,
    pub last_val: Option<Event>,
    pub finished: Option<Instant>,
}

/// A running or finished job.
pub struct Job {
    /// Its number, from 1.
    pub id: u64,
    /// The `ferrum-gpu` subcommand.
    pub cmd: String,
    /// Its arguments.
    pub args: Vec<String>,
    pub(crate) started: Instant,
    pub(crate) stop_file: Option<PathBuf>,
    pub(crate) inner: Mutex<Inner>,
    /// A stop was requested.
    pub(crate) stopping: AtomicBool,
    /// Kill the process now.
    pub(crate) kill_now: AtomicBool,
    /// When a polite stop turns into a kill.
    pub(crate) deadline: Mutex<Option<Instant>>,
    pub(crate) grace: Duration,
}

/// What a poll returns: the lines from `since` on.
pub struct Poll {
    /// `running`, `stopping`, `done`, `stopped` or `error`.
    pub state: &'static str,
    /// The exit status once the process has ended.
    pub exit: Option<i32>,
    /// Why the process could not be started or waited for.
    pub error: Option<String>,
    /// `(index, line)`.
    pub lines: Vec<(usize, Line)>,
    /// The `since` to ask for next time.
    pub next: usize,
    /// The latest training step report.
    pub last_step: Option<Event>,
    /// The latest validation report.
    pub last_val: Option<Event>,
    /// Seconds since the start (or the run time once finished).
    pub seconds: f64,
}

impl Job {
    pub(crate) fn new(
        id: u64,
        cmd: &str,
        args: Vec<String>,
        stop_file: Option<PathBuf>,
        grace: Duration,
    ) -> Job {
        Job {
            id,
            cmd: cmd.into(),
            args,
            started: Instant::now(),
            stop_file,
            inner: Mutex::new(Inner {
                lines: VecDeque::new(),
                base: 0,
                state: State::Running,
                last_step: None,
                last_val: None,
                finished: None,
            }),
            stopping: AtomicBool::new(false),
            kill_now: AtomicBool::new(false),
            deadline: Mutex::new(None),
            grace,
        }
    }

    /// Append an output line (parsing progress), dropping the oldest beyond [`MAX_LINES`].
    pub(crate) fn push(&self, err: bool, text: String) {
        let ev = progress::parse_line(&text);
        let mut i = lock(&self.inner);
        match &ev {
            Some(e @ Event::Step { .. }) => i.last_step = Some(e.clone()),
            Some(e @ Event::Val { .. }) => i.last_val = Some(e.clone()),
            None => {}
        }
        i.lines.push_back(Line { err, text, ev });
        if i.lines.len() > MAX_LINES {
            i.lines.pop_front();
            i.base += 1;
        }
    }

    pub(crate) fn finish(&self, state: State) {
        let mut i = lock(&self.inner);
        i.state = state;
        i.finished = Some(Instant::now());
    }

    /// The status word: `running`, `stopping`, `done`, `stopped` or `error`.
    pub fn word(&self) -> &'static str {
        let stopping = self.stopping.load(Ordering::SeqCst);
        match &lock(&self.inner).state {
            State::Running if stopping => "stopping",
            State::Running => "running",
            State::Ended(_) if stopping => "stopped",
            State::Ended(0) => "done",
            State::Ended(_) | State::Failed(_) => "error",
        }
    }

    /// True once the process has ended or could not start.
    pub fn is_finished(&self) -> bool {
        lock(&self.inner).state != State::Running
    }

    /// The lines from index `since` on, and the state.
    pub fn poll(&self, since: usize) -> Poll {
        let word = self.word();
        let i = lock(&self.inner);
        let from = since.max(i.base);
        let lines = i
            .lines
            .iter()
            .enumerate()
            .skip(from - i.base)
            .map(|(k, l)| (i.base + k, l.clone()))
            .collect();
        let (exit, error) = match &i.state {
            State::Ended(c) => (Some(*c), None),
            State::Failed(e) => (None, Some(e.clone())),
            State::Running => (None, None),
        };
        Poll {
            state: word,
            exit,
            error,
            lines,
            next: i.base + i.lines.len(),
            last_step: i.last_step.clone(),
            last_val: i.last_val.clone(),
            seconds: i
                .finished
                .unwrap_or_else(Instant::now)
                .duration_since(self.started)
                .as_secs_f64(),
        }
    }

    /// Ask the job to stop. A training command with a `<out>.stop` file is asked politely (the file
    /// is written, and the process is killed if it has not ended after the grace period); any other
    /// command, or `force`, is killed. False if the job had already finished.
    pub fn stop(&self, force: bool) -> bool {
        if self.is_finished() {
            return false;
        }
        self.stopping.store(true, Ordering::SeqCst);
        match (&self.stop_file, force) {
            (Some(f), false) if std::fs::write(f, b"stop requested by the Studio\n").is_ok() => {
                *lock(&self.deadline) = Some(Instant::now() + self.grace);
            }
            _ => self.kill_now.store(true, Ordering::SeqCst),
        }
        true
    }
}

/// All jobs.
pub struct Jobs {
    cli: Mutex<Option<PathBuf>>,
    next: AtomicU64,
    table: Mutex<BTreeMap<u64, Arc<Job>>>,
    grace: Duration,
}

impl Default for Jobs {
    fn default() -> Self {
        Jobs::new(None)
    }
}

impl Jobs {
    /// A job table; `cli` is an explicit `ferrum-gpu` path (else it is searched for on first use).
    pub fn new(cli: Option<PathBuf>) -> Jobs {
        Jobs::with_grace(cli, GRACE)
    }

    /// Like [`Jobs::new`] with another grace period (tests use a short one).
    pub fn with_grace(cli: Option<PathBuf>, grace: Duration) -> Jobs {
        Jobs {
            cli: Mutex::new(cli),
            next: AtomicU64::new(1),
            table: Mutex::new(BTreeMap::new()),
            grace,
        }
    }

    /// Set the explicit CLI path (the `--cli` flag).
    pub fn set_cli(&self, path: Option<PathBuf>) {
        *lock(&self.cli) = path;
    }

    /// Validate and start a job.
    pub fn start(&self, cmd: &str, args: Vec<String>) -> Result<Arc<Job>, String> {
        allow::validate(cmd, &args)?;
        let exe = cli::find_here(lock(&self.cli).as_deref())?;
        let running = lock(&self.table)
            .values()
            .filter(|j| !j.is_finished())
            .count();
        if running >= MAX_RUNNING {
            return Err(format!(
                "{MAX_RUNNING} jobs are already running; wait or stop one"
            ));
        }
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let job = Arc::new(Job::new(
            id,
            cmd,
            args.clone(),
            allow::stop_file(cmd, &args),
            self.grace,
        ));
        runner::spawn(&exe, job.clone())?;
        let mut t = lock(&self.table);
        t.insert(id, job.clone());
        while t.len() > MAX_KEPT {
            match t.iter().find(|(_, j)| j.is_finished()).map(|(k, _)| *k) {
                Some(k) => t.remove(&k),
                None => break,
            };
        }
        Ok(job)
    }

    /// A job by id.
    pub fn get(&self, id: u64) -> Option<Arc<Job>> {
        lock(&self.table).get(&id).cloned()
    }

    /// Every remembered job, oldest first.
    pub fn list(&self) -> Vec<Arc<Job>> {
        lock(&self.table).values().cloned().collect()
    }
}
