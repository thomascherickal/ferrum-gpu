//! Jobs: `ferrum-gpu` commands run **in this process**, on their own threads, with their output
//! streamed to whoever polls (the web page, the desktop window).
//!
//! A job runs exactly what the command line runs ([`crate::commands::run`]: an argument list in, a
//! report written to a `Write` out), so there is one implementation of every command. Output lines
//! are collected as they are written; progress lines are parsed into events ([`progress`]); a stop
//! raises the job's cancel flag, which the training and generation loops poll ([`crate::cancel`]).
//! A command that fails returns an error and a command that panics is caught: either ends the job,
//! never the application. What may run is fixed in [`allow`]; the JSON the page reads is in [`view`].

pub mod allow;
pub mod progress;
pub mod runner;
pub mod view;

#[cfg(test)]
mod tests;

use progress::Event;
use std::collections::{BTreeMap, VecDeque};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// Most output lines kept per job (older ones are dropped, and `since` stays valid).
pub const MAX_LINES: usize = 20_000;
/// Most jobs running at once.
pub const MAX_RUNNING: usize = 4;
/// Most jobs remembered; the oldest finished ones are forgotten first.
pub const MAX_KEPT: usize = 50;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A command runner: the argument list (command name first) and where to write the report.
/// The default is [`crate::commands::run`]; tests give a fake.
pub type RunFn = dyn Fn(&[String], &mut dyn Write) -> Result<(), String> + Send + Sync;

/// One line of a job's output.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// Came from the error path (a command that failed, or a panic).
    pub err: bool,
    /// The text, without the line ending.
    pub text: String,
    /// The progress event it carries, if it is a progress line.
    pub ev: Option<Event>,
}

/// Where a job stands.
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// The command is running.
    Running,
    /// The command ended; its exit status (0 = success, 2 = it returned an error, 101 = it panicked).
    Ended(i32),
}

pub(crate) struct Inner {
    pub lines: VecDeque<Line>,
    /// Lines dropped from the front so far.
    pub base: usize,
    pub state: State,
    pub last_step: Option<Event>,
    pub last_val: Option<Event>,
    pub last_eval: Option<Event>,
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
    pub(crate) inner: Mutex<Inner>,
    /// A stop was requested.
    pub(crate) stopping: AtomicBool,
    /// The flag the command's loops poll through [`crate::cancel`].
    pub(crate) cancel: Arc<AtomicBool>,
}

/// What a poll returns: the lines from `since` on.
pub struct Poll {
    /// `running`, `stopping`, `done`, `stopped` or `error`.
    pub state: &'static str,
    /// The exit status once the command has ended.
    pub exit: Option<i32>,
    /// `(index, line)`.
    pub lines: Vec<(usize, Line)>,
    /// The `since` to ask for next time.
    pub next: usize,
    /// The latest training step report.
    pub last_step: Option<Event>,
    /// The latest validation report.
    pub last_val: Option<Event>,
    /// The result of an `eval`, once it has printed.
    pub last_eval: Option<Event>,
    /// Seconds since the start (or the run time once finished).
    pub seconds: f64,
}

impl Job {
    pub(crate) fn new(id: u64, cmd: &str, args: Vec<String>) -> Job {
        Job {
            id,
            cmd: cmd.into(),
            args,
            started: Instant::now(),
            inner: Mutex::new(Inner {
                lines: VecDeque::new(),
                base: 0,
                state: State::Running,
                last_step: None,
                last_val: None,
                last_eval: None,
                finished: None,
            }),
            stopping: AtomicBool::new(false),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Append an output line (parsing progress), dropping the oldest beyond [`MAX_LINES`].
    pub(crate) fn push(&self, err: bool, text: String) {
        let ev = progress::parse_line(&text);
        let mut i = lock(&self.inner);
        match &ev {
            Some(e @ Event::Step { .. }) => i.last_step = Some(e.clone()),
            Some(e @ Event::Val { .. }) => i.last_val = Some(e.clone()),
            Some(e @ Event::Eval { .. }) => i.last_eval = Some(e.clone()),
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
            State::Ended(_) => "error",
        }
    }

    /// True once the command has ended.
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
        let exit = match &i.state {
            State::Ended(c) => Some(*c),
            State::Running => None,
        };
        Poll {
            state: word,
            exit,
            lines,
            next: i.base + i.lines.len(),
            last_step: i.last_step.clone(),
            last_val: i.last_val.clone(),
            last_eval: i.last_eval.clone(),
            seconds: i
                .finished
                .unwrap_or_else(Instant::now)
                .duration_since(self.started)
                .as_secs_f64(),
        }
    }

    /// Ask the job to stop. Cooperative: training and generation notice at the next step or token
    /// and end cleanly (a training run writes the model it has so far); a command without such a
    /// check (prepare, tokenize, the exports) runs to its end. False if the job had already finished.
    pub fn stop(&self) -> bool {
        if self.is_finished() {
            return false;
        }
        self.stopping.store(true, Ordering::SeqCst);
        self.cancel.store(true, Ordering::SeqCst);
        true
    }
}

/// All jobs.
pub struct Jobs {
    next: AtomicU64,
    table: Mutex<BTreeMap<u64, Arc<Job>>>,
    run: Arc<RunFn>,
}

impl Default for Jobs {
    fn default() -> Self {
        Jobs::new()
    }
}

impl Jobs {
    /// A job table whose jobs run the real commands.
    pub fn new() -> Jobs {
        Jobs::with_runner(Arc::new(|args, out| crate::commands::run(args, out)))
    }

    /// A job table with another command runner (tests use fakes).
    pub fn with_runner(run: Arc<RunFn>) -> Jobs {
        Jobs {
            next: AtomicU64::new(1),
            table: Mutex::new(BTreeMap::new()),
            run,
        }
    }

    /// Validate and start a job.
    pub fn start(&self, cmd: &str, args: Vec<String>) -> Result<Arc<Job>, String> {
        allow::validate(cmd, &args)?;
        let mut t = lock(&self.table);
        let running: Vec<_> = t.values().filter(|j| !j.is_finished()).collect();
        if running.len() >= MAX_RUNNING {
            return Err(format!(
                "{MAX_RUNNING} jobs are already running; wait or stop one"
            ));
        }
        // adapter enumeration is not safe from several threads at once, so the commands that open a
        // device take turns
        if allow::uses_device(cmd) {
            if let Some(other) = running.iter().find(|j| allow::uses_device(&j.cmd)) {
                return Err(format!(
                    "job {} ({}) is using the device; wait for it or stop it first",
                    other.id, other.cmd
                ));
            }
        }
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let job = Arc::new(Job::new(id, cmd, args));
        runner::spawn(self.run.clone(), job.clone())?;
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
