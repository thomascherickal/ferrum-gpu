use super::*;

impl Gpu {
    /// Every adapter wgpu can see (empty on a machine without drivers).
    pub fn list() -> Vec<AdapterSummary> {
        let inst = wgpu::Instance::default();
        block_on(inst.enumerate_adapters(wgpu::Backends::all()))
            .iter()
            .map(summarize)
            .collect()
    }

    /// Open an adapter chosen by [`choose`] (`name` is a case-insensitive fragment). The device
    /// is requested with the adapter's own limits, so nothing is assumed beyond what it reports.
    pub fn open(name: Option<&str>) -> Result<Gpu, GpuError> {
        let inst = wgpu::Instance::default();
        let mut adapters = block_on(inst.enumerate_adapters(wgpu::Backends::all()));
        let summaries: Vec<_> = adapters.iter().map(summarize).collect();
        let idx = choose(&summaries, name)?;
        let adapter = adapters.swap_remove(idx);
        let limits = adapter.limits();
        let desc = wgpu::DeviceDescriptor {
            label: Some("ferrum"),
            required_limits: limits.clone(),
            ..Default::default()
        };
        let requested = block_on(adapter.request_device(&desc));
        let (device, queue) = stage(requested, "request_device failed")?;
        // Device loss and uncaptured errors are recorded, never allowed to abort the process.
        let health = Arc::new(Mutex::new(None));
        let h = Arc::clone(&health);
        device.set_device_lost_callback(move |reason, msg| {
            *h.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(format!("GPU device lost: {reason:?}: {msg}"));
        });
        let h = Arc::clone(&health);
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            *h.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("GPU error: {e}"));
        }));
        Ok(Gpu {
            device,
            queue,
            info: summaries[idx].clone(),
            max_workgroups: limits.max_compute_workgroups_per_dimension,
            health,
            pipelines: Mutex::new(HashMap::new()),
            pipelines_built: AtomicUsize::new(0),
            buffers_created: AtomicUsize::new(0),
            batch: Mutex::new(None),
            #[cfg(any(test, feature = "fault-injection"))]
            faults: Faults::default(),
        })
    }

    /// Facts about the opened adapter.
    pub fn info(&self) -> &AdapterSummary {
        &self.info
    }

    /// Largest workgroup count per dispatch dimension.
    pub fn max_workgroups(&self) -> u32 {
        self.max_workgroups
    }
}

/// Poll until the asynchronous device-lost callback has landed (bounded, never panics).
#[cfg(test)]
pub(crate) fn wait_for_loss(g: &Gpu) {
    // Always nap after polling, so every line runs whether or not the loss has landed yet.
    (0..200).any(|_| {
        let _ = g.device.poll(wgpu::PollType::Poll);
        let lost = g.check().is_err();
        std::thread::sleep(std::time::Duration::from_millis(5));
        lost
    });
}

/// Open a GPU for a test, or print why not and return `None`. `FERRUM_GPU_ADAPTER` names an
/// adapter (e.g. `llvmpipe`); otherwise a hardware adapter is used when one exists.
pub fn open_for_test() -> Option<Gpu> {
    let want = std::env::var("FERRUM_GPU_ADAPTER").ok();
    match Gpu::open(want.as_deref()) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP: no usable GPU adapter ({e})");
            None
        }
    }
}
