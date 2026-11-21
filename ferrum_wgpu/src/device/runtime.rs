//! Device runtime shared by every kernel: error scopes, device-health, a pipeline cache, one
//! generic dispatch path, and plan-then-allocate.

use crate::budget::{Budget, BufferPlan};
use crate::device::{block_on, Gpu, GpuError};
use crate::tensor::GpuBuf;
use std::sync::atomic::Ordering;

/// Turn a wgpu error into our error type.
pub(crate) fn map_error(e: wgpu::Error) -> GpuError {
    match e {
        wgpu::Error::OutOfMemory { .. } => GpuError("GPU out of memory".into()),
        wgpu::Error::Validation { description, .. } => {
            GpuError(format!("GPU validation error: {description}"))
        }
        wgpu::Error::Internal { description, .. } => {
            GpuError(format!("GPU internal error: {description}"))
        }
    }
}

/// Workgroup counts `(x, y)` covering `work = (columns, rows)` with `tile = (columns, rows)`
/// per workgroup, or an error naming the per-dimension limit.
pub fn groups_for(
    work: (usize, usize),
    tile: (u32, u32),
    max: u32,
) -> Result<(u32, u32), GpuError> {
    let x = work.0.div_ceil(tile.0.max(1) as usize);
    let y = work.1.div_ceil(tile.1.max(1) as usize);
    if x > max as usize || y > max as usize {
        return Err(GpuError(format!(
            "dispatch of {}x{} needs {x}x{y} workgroups, adapter allows {max} per dimension",
            work.0, work.1
        )));
    }
    Ok((x as u32, y as u32))
}

/// Workgroups along x for 1-D dispatches. Fixed at the WebGPU baseline limit (not the adapter's
/// larger one) so [`gid`](PRELUDE) can use a constant stride.
pub const GROUPS_X: usize = 65535;
/// Invocations per 1-D workgroup.
pub const WG_1D: usize = 64;

/// WGSL prepended to every shader: the global index of a 1-D kernel dispatched with
/// [`groups_1d`] (rows of `GROUPS_X * WG_1D` invocations stacked along y).
pub const PRELUDE: &str = "fn gid(g: vec3<u32>) -> u32 { return g.y * 4194240u + g.x; }";

/// Workgroup counts `(x, y)` for a 1-D kernel over `n` elements with 64-wide workgroups,
/// spilling into y beyond 65,535 groups. Errors only past `65535 * 65535 * 64` elements.
pub fn groups_1d(n: usize) -> Result<(u32, u32), GpuError> {
    let g = n.div_ceil(WG_1D).max(1);
    let x = g.min(GROUPS_X);
    let y = g.div_ceil(x);
    if y > GROUPS_X {
        return Err(GpuError(format!(
            "{n} elements exceed the 1-D dispatch limit"
        )));
    }
    Ok((x as u32, y as u32))
}

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

    /// [`dispatch`](Self::dispatch) for a 1-D kernel over `n` elements (the groups come from
    /// [`groups_1d`]; an `n` beyond the 1-D limit is an error). Kept as one method so the
    /// oversize check exists once, not at every call site.
    pub fn dispatch_1d(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&GpuBuf],
        uniform: &[u32],
        n: usize,
    ) -> Result<(), GpuError> {
        self.dispatch(pipeline, buffers, uniform, groups_1d(n)?)
    }

    /// Record and submit one dispatch: `buffers` bind at 0..n, then a uniform of `uniform`
    /// u32 values (padded to 16 bytes) at binding n. Does not wait.
    pub fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&GpuBuf],
        uniform: &[u32],
        groups: (u32, u32),
    ) -> Result<(), GpuError> {
        let mut words = uniform.to_vec();
        words.resize(words.len().div_ceil(4).max(1) * 4, 0);
        self.record(|enc| {
            let ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (words.len() * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&ubuf, 0, bytemuck::cast_slice(&words));
            let mut entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.buf.as_entire_binding(),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: buffers.len() as u32,
                resource: ubuf.as_entire_binding(),
            });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        })
    }

    /// Run `f` against a command encoder. Outside a batch it is its own scoped submit; inside
    /// one the commands join the batch encoder, and any error surfaces at [`Gpu::end_batch`].
    fn record(&self, f: impl FnOnce(&mut wgpu::CommandEncoder)) -> Result<(), GpuError> {
        self.check()?;
        #[cfg(test)]
        self.injected_fault()?;
        {
            let mut batch = self.batch.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(enc) = batch.as_mut() {
                f(enc);
                return Ok(());
            }
        }
        self.scoped(|| {
            let mut enc = self.device.create_command_encoder(&Default::default());
            f(&mut enc);
            self.queue.submit([enc.finish()]);
        })
    }

    /// Test hook: let `n` more commands (dispatches and copies) succeed, then fail every later
    /// one with "injected fault". `usize::MAX` turns injection off. Used to drive the error
    /// branch of every `?` in code that issues many commands.
    #[cfg(test)]
    pub(crate) fn fail_after(&self, n: usize) {
        self.fault_after.store(n, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn injected_fault(&self) -> Result<(), GpuError> {
        match self.fault_after.load(Ordering::Relaxed) {
            usize::MAX => Ok(()),
            0 => Err(GpuError("injected fault".into())),
            n => {
                self.fault_after.store(n - 1, Ordering::Relaxed);
                Ok(())
            }
        }
    }

    /// Test helper: run `f` repeatedly, failing command 0, then 1, then 2, ... until it
    /// succeeds. Every failed run must report the injected fault and leave no batch open.
    /// Returns how many commands `f` issues, and turns injection off again.
    #[cfg(test)]
    pub(crate) fn sweep_faults(&self, mut f: impl FnMut() -> Result<(), GpuError>) -> usize {
        for k in 0..100_000 {
            self.fail_after(k);
            match f() {
                Err(e) => {
                    assert!(e.to_string().contains("injected fault"), "command {k}: {e}");
                    assert!(
                        !self.batching(),
                        "a batch was left open after a fault at command {k}"
                    );
                }
                Ok(()) => {
                    self.fail_after(usize::MAX);
                    return k;
                }
            }
        }
        panic!("the operation never succeeded");
    }

    /// Start recording every following dispatch and copy into one command encoder (one submit
    /// instead of one per kernel). Reads are refused until [`Gpu::end_batch`].
    pub fn begin_batch(&self) -> Result<(), GpuError> {
        self.check()?;
        let mut batch = self.batch.lock().unwrap_or_else(|e| e.into_inner());
        if batch.is_some() {
            return Err(GpuError("a batch is already open".into()));
        }
        *batch = Some(self.device.create_command_encoder(&Default::default()));
        Ok(())
    }

    /// Close the batch. With `submit` the recorded work runs; without it the work is dropped.
    /// Errors raised while recording are reported here.
    pub fn end_batch(&self, submit: bool) -> Result<(), GpuError> {
        let enc = self
            .batch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| GpuError("no batch is open".into()))?;
        if submit {
            self.queue.submit([enc.finish()]);
        }
        self.check()
    }

    /// True while a batch is recording.
    pub fn batching(&self) -> bool {
        self.batch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Copy `n` f32 from `src[src_off..]` to `dst[dst_off..]` on the device (no host round trip).
    pub fn copy_f32(
        &self,
        src: &GpuBuf,
        src_off: usize,
        dst: &GpuBuf,
        dst_off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        if src_off + n > src.len() || dst_off + n > dst.len() {
            return Err(GpuError(format!(
                "copy of {n} elements at {src_off}->{dst_off} exceeds the buffers ({} and {})",
                src.len(),
                dst.len()
            )));
        }
        if n == 0 {
            return Ok(());
        }
        self.record(|enc| {
            enc.copy_buffer_to_buffer(
                &src.buf,
                (src_off * 4) as u64,
                &dst.buf,
                (dst_off * 4) as u64,
                (n * 4) as u64,
            );
        })
    }

    /// Check a plan against the adapter's binding limit and `budget`, then (only if it fits)
    /// allocate one zeroed buffer per row, in order. Rows are rounded up to whole f32s.
    pub fn alloc_plan(
        &self,
        plan: &BufferPlan,
        budget: Option<Budget>,
    ) -> Result<Vec<GpuBuf>, GpuError> {
        plan.check(self.info().max_binding, budget)
            .map_err(|e| GpuError(e.to_string()))?;
        plan.rows()
            .iter()
            .map(|(_, bytes)| self.zeros(bytes.div_ceil(4) as usize))
            .collect()
    }
}
#[cfg(test)]
mod tests;
