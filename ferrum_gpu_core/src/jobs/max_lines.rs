use super::*;

/// Most output lines kept per job (older ones are dropped, and `since` stays valid).
pub const MAX_LINES: usize = 20_000;
/// Most jobs running at once.
pub const MAX_RUNNING: usize = 4;
/// Most jobs remembered; the oldest finished ones are forgotten first.
pub const MAX_KEPT: usize = 50;

pub(super) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
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
