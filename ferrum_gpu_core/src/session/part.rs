use super::*;

impl Session {
    /// Open the device for `choice`. `adapter` names a GPU (fragment). `Auto` falls back to the CPU
    /// with a note; `Gpu` with nothing usable is an error.
    pub fn new(
        choice: DeviceChoice,
        adapter: Option<&str>,
        cpu_threads: usize,
    ) -> Result<Session, String> {
        Session::new_with(choice, adapter, cpu_threads, None)
    }

    /// [`new`](Session::new) with the weight storage of a GPU decoder chosen.
    pub fn new_with(
        choice: DeviceChoice,
        adapter: Option<&str>,
        cpu_threads: usize,
        weights: Option<QKind>,
    ) -> Result<Session, String> {
        let adapters = Gpu::list();
        let mut note = None;
        let backend: Arc<dyn Backend> = match choice {
            DeviceChoice::Cpu => Arc::new(CpuBackend::new(cpu_threads)),
            DeviceChoice::Gpu => Arc::new(GpuBackend::open(adapter)?.with_weights(weights)),
            DeviceChoice::Auto => match GpuBackend::open(adapter) {
                Ok(g) => Arc::new(g.with_weights(weights)),
                Err(e) => {
                    note = Some(format!("no usable GPU ({e}); using the CPU"));
                    Arc::new(CpuBackend::new(cpu_threads))
                }
            },
        };
        let mut s = Session::with_backend(backend, choice, note, adapters);
        s.weights = weights;
        Ok(s)
    }

    /// A session around an already-built backend (tests use this with fakes).
    pub fn with_backend(
        backend: Arc<dyn Backend>,
        requested: DeviceChoice,
        fallback_note: Option<String>,
        adapters: Vec<AdapterSummary>,
    ) -> Session {
        Session {
            backend,
            requested,
            fallback_note,
            adapters,
            weights: None,
        }
    }
}
