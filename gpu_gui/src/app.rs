//! Shared application state: the chosen backend, the current model and the training job.

use crate::engine::{preset, Backend, CpuBackend, GpuBackend, Progress, TrainSpec};
use ferrum_ref::llama::Llama;
use ferrum_wgpu::device::select::AdapterSummary;
use ferrum_wgpu::Gpu;
use std::sync::{Arc, Mutex, MutexGuard};

/// Which device the user asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The CPU oracle.
    Cpu,
    /// A GPU adapter; an error if none can be opened.
    Gpu,
    /// A GPU if one opens, else the CPU.
    Auto,
}

impl DeviceChoice {
    /// Parse `cpu`, `gpu` or `auto`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            other => Err(format!("unknown device '{other}' (cpu, gpu, auto)")),
        }
    }

    /// The flag spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Auto => "auto",
        }
    }
}

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
    /// Where models run.
    pub backend: Arc<dyn Backend>,
    /// What the user asked for.
    pub requested: DeviceChoice,
    /// Why the backend is not the requested one, if it is not.
    pub fallback_note: Option<String>,
    /// Every adapter the machine reports.
    pub adapters: Vec<AdapterSummary>,
    /// The current model.
    pub model: Mutex<Option<Llama>>,
    /// The training job.
    pub job: Mutex<Job>,
    /// The last self-test result posted by a browser (Phase 5b).
    pub selftest: Mutex<Option<String>>,
}

/// Lock, recovering the data if a worker panicked while holding the lock.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl App {
    /// Build the state for `choice`. `adapter` names a GPU (fragment). `Auto` falls back to the
    /// CPU with a note; `Gpu` with nothing usable is an error.
    pub fn new(
        choice: DeviceChoice,
        adapter: Option<&str>,
        cpu_threads: usize,
    ) -> Result<App, String> {
        let adapters = Gpu::list();
        let mut note = None;
        let backend: Arc<dyn Backend> = match choice {
            DeviceChoice::Cpu => Arc::new(CpuBackend::new(cpu_threads)),
            DeviceChoice::Gpu => Arc::new(GpuBackend::open(adapter)?),
            DeviceChoice::Auto => match GpuBackend::open(adapter) {
                Ok(g) => Arc::new(g),
                Err(e) => {
                    note = Some(format!("no usable GPU ({e}); using the CPU"));
                    Arc::new(CpuBackend::new(cpu_threads))
                }
            },
        };
        Ok(App::with_backend(backend, choice, note, adapters))
    }

    /// State around an already-built backend (tests use this with fakes).
    pub fn with_backend(
        backend: Arc<dyn Backend>,
        requested: DeviceChoice,
        fallback_note: Option<String>,
        adapters: Vec<AdapterSummary>,
    ) -> App {
        App {
            backend,
            requested,
            fallback_note,
            adapters,
            model: Mutex::new(None),
            job: Mutex::new(Job {
                state: JobState::Idle,
                progress: Arc::new(Mutex::new(Progress::default())),
            }),
            selftest: Mutex::new(None),
        }
    }

    /// Start a training run on a copy of the current model. Errors if there is no model, a run
    /// is already going, or the spec does not fit the model.
    pub fn start_training(self: &Arc<Self>, spec: TrainSpec) -> Result<(), String> {
        let model = lock(&self.model)
            .clone()
            .ok_or("create or load a model first")?;
        let progress = {
            let mut job = lock(&self.job);
            if job.state == JobState::Running {
                return Err("a training run is already in progress".into());
            }
            spec.validate(&model.cfg)?;
            job.state = JobState::Running;
            job.progress = Arc::new(Mutex::new(Progress::default()));
            job.progress.clone()
        };
        let app = self.clone();
        std::thread::spawn(move || {
            let result = app.backend.train(&model, &spec, &progress);
            let stopped = lock(&progress).stop;
            let mut job = lock(&app.job);
            job.state = match result {
                Ok(w) => {
                    *lock(&app.model) = Some(w);
                    if stopped {
                        JobState::Stopped
                    } else {
                        JobState::Done
                    }
                }
                Err(e) => JobState::Failed(e),
            };
        });
        Ok(())
    }

    /// Ask a running job to stop after its current step.
    pub fn stop_training(&self) -> bool {
        let job = lock(&self.job);
        if job.state == JobState::Running {
            lock(&job.progress).stop = true;
            true
        } else {
            false
        }
    }

    /// Name of the preset whose configuration equals the model's, if any.
    pub fn preset_of(model: &Llama) -> Option<&'static str> {
        ["tiny", "small", "medium"]
            .into_iter()
            .find(|p| preset(p).as_ref() == Some(&model.cfg))
    }
}
