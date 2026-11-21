use super::*;

impl Gpu {
    /// Fail if the device was lost (sticky: it never recovers) or reported an uncaptured error
    /// since the last check (reported once, then cleared: a bad dispatch does not brick the device).
    pub fn check(&self) -> Result<(), GpuError> {
        let mut h = self.health.lock().unwrap_or_else(|e| e.into_inner());
        match h.as_ref() {
            Some(msg) if msg.starts_with("GPU device lost") => Err(GpuError(msg.clone())),
            Some(_) => Err(GpuError(h.take().expect("matched Some"))),
            None => Ok(()),
        }
    }

    /// Destroy the device (it reports lost afterwards). Mainly for tests and shutdown.
    pub fn destroy(&self) {
        self.device.destroy();
    }

    /// Run `f` with validation and out-of-memory scopes active; errors become `Err`.
    pub(crate) fn scoped<T>(&self, f: impl FnOnce() -> T) -> Result<T, GpuError> {
        self.check()?;
        let v = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let o = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let out = f();
        let oom = block_on(o.pop());
        let val = block_on(v.pop());
        match oom.or(val) {
            Some(e) => Err(map_error(e)),
            None => Ok(out),
        }
    }

    /// A compute pipeline for `wgsl` (entry point `main`), compiled once per `key`.
    pub fn pipeline(
        &self,
        key: &'static str,
        wgsl: &'static str,
    ) -> Result<wgpu::ComputePipeline, GpuError> {
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(key) {
            return Ok(p.clone());
        }
        let p = self.scoped(|| {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(key),
                    source: wgpu::ShaderSource::Wgsl(format!("{PRELUDE}\n{wgsl}").into()),
                });
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(key),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        })?;
        self.pipelines_built.fetch_add(1, Ordering::Relaxed);
        cache.insert(key, p.clone());
        Ok(p)
    }

    /// Pipelines compiled so far (cache misses).
    pub fn pipelines_built(&self) -> usize {
        self.pipelines_built.load(Ordering::Relaxed)
    }

    /// Buffers created so far.
    pub fn buffers_created(&self) -> usize {
        self.buffers_created.load(Ordering::Relaxed)
    }
}
