use super::*;

impl App {
    /// Build the state for `choice`. `adapter` names a GPU (fragment). `Auto` falls back to the
    /// CPU with a note; `Gpu` with nothing usable is an error.
    pub fn new(
        choice: DeviceChoice,
        adapter: Option<&str>,
        cpu_threads: usize,
    ) -> Result<App, String> {
        let s = Session::new(choice, adapter, cpu_threads)?;
        Ok(App::with_session(s))
    }

    /// State around an already-built backend (tests use this with fakes).
    pub fn with_backend(
        backend: Arc<dyn Backend>,
        requested: DeviceChoice,
        fallback_note: Option<String>,
        adapters: Vec<AdapterSummary>,
    ) -> App {
        App::with_session(Session::with_backend(
            backend,
            requested,
            fallback_note,
            adapters,
        ))
    }

    /// This state with another job table (tests give one whose commands are fakes).
    pub fn with_jobs(mut self, jobs: crate::jobs::Jobs) -> App {
        self.jobs = jobs;
        self
    }

    /// State around an opened [`Session`].
    pub fn with_session(session: Session) -> App {
        App {
            session,
            model: Mutex::new(None),
            version: AtomicU64::new(0),
            job: Mutex::new(Job {
                state: JobState::Idle,
                progress: Arc::new(Mutex::new(Progress::default())),
            }),
            selftest: Mutex::new(None),
            jobs: crate::jobs::Jobs::default(),
            serve: ferrum_gpu_core::serving::host::ServeSlot::new(),
            max_params: crate::api::MAX_PARAMS,
        }
    }
}
