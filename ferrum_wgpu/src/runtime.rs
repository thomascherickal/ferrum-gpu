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
    /// Fail if the device was lost or reported an uncaptured error since it was opened.
    pub fn check(&self) -> Result<(), GpuError> {
        match self
            .health
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            Some(msg) => Err(GpuError(format!("GPU device lost: {msg}"))),
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
        self.scoped(|| {
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
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, 1);
            }
            self.queue.submit([enc.finish()]);
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
