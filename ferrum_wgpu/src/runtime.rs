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
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
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
mod tests {
    use super::*;
    use crate::budget::Source;
    use crate::device::open_for_test;

    #[test]
    fn groups_cover_and_respect_the_limit() {
        assert_eq!(groups_for((17, 9), (8, 8), 65535).unwrap(), (3, 2));
        assert_eq!(groups_for((0, 0), (8, 8), 65535).unwrap(), (0, 0));
        assert_eq!(groups_for((8, 8), (0, 0), 65535).unwrap(), (8, 8)); // zero tile is clamped
        let e = groups_for((8, 8 * 70_000), (8, 8), 65535).unwrap_err();
        assert!(e.to_string().contains("65535"));
    }

    #[test]
    fn wgpu_errors_map_to_readable_messages() {
        let src = || -> wgpu::ErrorSource { "x".into() };
        assert!(map_error(wgpu::Error::OutOfMemory { source: src() })
            .to_string()
            .contains("out of memory"));
        let v = map_error(wgpu::Error::Validation {
            source: src(),
            description: "bad bind".into(),
        });
        assert!(v.to_string().contains("bad bind"));
        let i = map_error(wgpu::Error::Internal {
            source: src(),
            description: "oops".into(),
        });
        assert!(i.to_string().contains("oops"));
    }

    #[test]
    fn pipelines_are_compiled_once_per_key() {
        let Some(g) = open_for_test() else { return };
        let src = "@group(0) @binding(0) var<storage, read_write> o: array<f32>; @compute @workgroup_size(1) fn main() { o[0] = 1.0; }";
        let before = g.pipelines_built();
        g.pipeline("test-once", src).unwrap();
        g.pipeline("test-once", src).unwrap();
        assert_eq!(g.pipelines_built(), before + 1);
    }

    #[test]
    fn a_broken_shader_is_an_error_not_a_panic() {
        let Some(g) = open_for_test() else { return };
        let e = g
            .pipeline("test-broken", "this is not wgsl")
            .expect_err("must fail");
        assert!(
            e.to_string().contains("validation") || e.to_string().contains("internal"),
            "{e}"
        );
    }

    #[test]
    fn dispatch_runs_a_kernel_through_the_generic_path() {
        let Some(g) = open_for_test() else { return };
        let src = "struct U { v: u32, p0: u32, p1: u32, p2: u32 }
            @group(0) @binding(0) var<storage, read_write> o: array<f32>;
            @group(0) @binding(1) var<uniform> u: U;
            @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) i: vec3<u32>) { o[i.x] = f32(u.v + i.x); }";
        let p = g.pipeline("test-dispatch", src).unwrap();
        let out = g.zeros(4).unwrap();
        g.dispatch(&p, &[&out], &[10], (4, 1)).unwrap();
        assert_eq!(g.read(&out).unwrap(), vec![10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn a_plan_over_budget_allocates_nothing() {
        let Some(g) = open_for_test() else { return };
        let mut plan = BufferPlan::default();
        plan.add("a", 4096);
        plan.add("b", 4096);
        let before = g.buffers_created();
        let tight = Some(Budget {
            bytes: 5000,
            source: Source::Override,
        });
        let e = g.alloc_plan(&plan, tight).expect_err("must refuse");
        assert!(e.to_string().contains("8192") && e.to_string().contains("5000"));
        assert_eq!(
            g.buffers_created(),
            before,
            "no buffer may be created on refusal"
        );
    }

    #[test]
    fn a_batch_runs_commands_in_order_in_one_submit() {
        let Some(g) = open_for_test() else { return };
        let (a, b, c) = (
            g.upload(&[1.0, 2.0, 3.0]).unwrap(),
            g.zeros(3).unwrap(),
            g.zeros(3).unwrap(),
        );
        assert!(!g.batching());
        g.begin_batch().unwrap();
        assert!(g.batching());
        // c depends on b, which depends on a: only in-order execution gives c == a.
        g.copy_f32(&a, 0, &b, 0, 3).unwrap();
        g.copy_f32(&b, 0, &c, 0, 3).unwrap();
        assert!(g.read(&c).unwrap_err().to_string().contains("batch"));
        g.end_batch(true).unwrap();
        assert!(!g.batching());
        assert_eq!(g.read(&c).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_discarded_batch_does_nothing_and_misuse_is_an_error() {
        let Some(g) = open_for_test() else { return };
        let (a, b) = (g.upload(&[5.0]).unwrap(), g.zeros(1).unwrap());
        g.begin_batch().unwrap();
        g.copy_f32(&a, 0, &b, 0, 1).unwrap();
        assert!(g
            .begin_batch()
            .unwrap_err()
            .to_string()
            .contains("already open"));
        g.end_batch(false).unwrap();
        assert_eq!(
            g.read(&b).unwrap(),
            vec![0.0],
            "discarded work must not run"
        );
        assert!(g
            .end_batch(true)
            .unwrap_err()
            .to_string()
            .contains("no batch"));
        // A dispatch inside a batch goes through the same generic path.
        let src = "struct U { v: u32, p0: u32, p1: u32, p2: u32 }
            @group(0) @binding(0) var<storage, read_write> o: array<f32>;
            @group(0) @binding(1) var<uniform> u: U;
            @compute @workgroup_size(1) fn main() { o[0] = f32(u.v); }";
        let p = g.pipeline("test-batch-dispatch", src).unwrap();
        let out = g.zeros(1).unwrap();
        g.begin_batch().unwrap();
        g.dispatch(&p, &[&out], &[7], (1, 1)).unwrap();
        g.end_batch(true).unwrap();
        assert_eq!(g.read(&out).unwrap(), vec![7.0]);
    }

    #[test]
    fn a_bad_batched_dispatch_fails_at_end_batch_and_does_not_brick_the_device() {
        let Some(g) = open_for_test() else { return };
        // This shader declares no uniform, but dispatch always binds one: the bind group is invalid.
        let bad = "@group(0) @binding(0) var<storage, read_write> o: array<f32>; @compute @workgroup_size(1) fn main() { o[0] = 1.0; }";
        let p = g.pipeline("test-batch-bad", bad).unwrap();
        let out = g.zeros(1).unwrap();
        g.begin_batch().unwrap();
        g.dispatch(&p, &[&out], &[1], (1, 1)).unwrap(); // recording does not validate
        let e = g.end_batch(true).unwrap_err();
        assert!(e.to_string().contains("GPU error"), "{e}");
        // The error was reported once; the device keeps working.
        g.check().unwrap();
        assert_eq!(g.read(&g.upload(&[3.0]).unwrap()).unwrap(), vec![3.0]);
        // Device loss, by contrast, is sticky.
        g.destroy();
        for _ in 0..200 {
            let _ = g.device.poll(wgpu::PollType::Poll);
            if g.check().is_err() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(g.check().unwrap_err().to_string().contains("device lost"));
        assert!(
            g.check().unwrap_err().to_string().contains("device lost"),
            "still lost on the second check"
        );
    }

    #[test]
    fn copy_f32_moves_a_slice_on_the_device_and_checks_bounds() {
        let Some(g) = open_for_test() else { return };
        let src = g.upload(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let dst = g.upload(&[9.0; 6]).unwrap();
        g.copy_f32(&src, 1, &dst, 2, 3).unwrap();
        assert_eq!(g.read(&dst).unwrap(), vec![9.0, 9.0, 2.0, 3.0, 4.0, 9.0]);
        g.copy_f32(&src, 0, &dst, 0, 0).unwrap();
        assert!(g
            .copy_f32(&src, 3, &dst, 0, 3)
            .unwrap_err()
            .to_string()
            .contains("exceeds"));
        assert!(g.copy_f32(&src, 0, &dst, 5, 2).is_err());
    }

    #[test]
    fn a_plan_that_fits_allocates_one_buffer_per_row() {
        let Some(g) = open_for_test() else { return };
        let mut plan = BufferPlan::default();
        plan.add("a", 16);
        plan.add("b", 6); // rounds up to 2 f32
        let bufs = g.alloc_plan(&plan, None).unwrap();
        assert_eq!(bufs.iter().map(|b| b.len()).collect::<Vec<_>>(), vec![4, 2]);
    }

    #[test]
    fn a_destroyed_device_reports_lost_instead_of_panicking() {
        let Some(g) = open_for_test() else { return };
        g.check().unwrap();
        g.destroy();
        // The lost callback fires asynchronously; poll until it lands (bounded).
        for _ in 0..200 {
            let _ = g.device.poll(wgpu::PollType::Poll);
            if g.check().is_err() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(g.check().is_err(), "device loss must be observed");
        let e = g.zeros(4).expect_err("use after loss must fail");
        assert!(e.to_string().contains("lost"), "{e}");
    }

    #[test]
    fn a_huge_allocation_is_an_error_or_succeeds_but_never_panics() {
        let Some(g) = open_for_test() else { return };
        let n = (g.info().max_binding / 4) as usize;
        match g.zeros(n) {
            Ok(b) => assert_eq!(b.len(), n),
            Err(e) => eprintln!("allocation refused as expected: {e}"),
        }
        g.check().unwrap();
    }
}
