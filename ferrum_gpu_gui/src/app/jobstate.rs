use super::*;

/// Where a training run stands.
#[derive(Clone, Debug, PartialEq)]
pub enum JobState {
    /// Nothing has run yet.
    Idle,
    /// A run is in progress.
    Running,
    /// The last run finished all its steps.
    Done,
    /// The last run was stopped early (its partial weights were kept).
    Stopped,
    /// The last run failed.
    Failed(String),
}

impl JobState {
    /// The status word the API reports.
    pub fn word(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "running",
            Self::Done => "done",
            Self::Stopped => "stopped",
            Self::Failed(_) => "error",
        }
    }
}

/// The training job slot.
pub struct Job {
    /// State of the latest run.
    pub state: JobState,
    /// Its progress, shared with the worker thread.
    pub progress: Arc<Mutex<Progress>>,
}

/// Everything the routes share.
pub struct App {
    /// The opened device: `app.backend`, `app.requested`, `app.fallback_note` and `app.adapters` reach
    /// it through `Deref`.
    pub session: Session,
    /// The current model. Replace it with [`App::set_model`] so [`App::version`] moves.
    pub model: Mutex<Option<Llama>>,
    /// Bumped whenever the model is replaced (the browser runtime re-downloads on a change).
    pub version: AtomicU64,
    /// The training job.
    pub job: Mutex<Job>,
    /// The last self-test result posted by a browser (Phase 5b).
    pub selftest: Mutex<Option<String>>,
    /// `ferrum-gpu` commands run as in-process jobs (the Studio tabs).
    pub jobs: crate::jobs::Jobs,
    /// The JSON server the Serve tab can run in this process.
    pub serve: ferrum_gpu_core::serving::host::ServeSlot,
    /// Most parameters a model imported over HTTP may have (a larger one is refused with 413).
    pub max_params: usize,
}

/// Lock, recovering the data if a worker panicked while holding the lock.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl std::ops::Deref for App {
    type Target = Session;
    fn deref(&self) -> &Session {
        &self.session
    }
}
