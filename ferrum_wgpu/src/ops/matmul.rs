//! Matmul kernels `C[m,n] = A[m,k] * B[k,n]` on the GPU: three variants for the Phase 0 spike.

use crate::device::{Gpu, GpuError};
use crate::runtime::groups_for;
use crate::tensor::GpuBuf;

/// Kernel variants compared in the spike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// One invocation per output element.
    Naive,
    /// 16x16 shared-memory tiles.
    Tiled16,
    /// 64x64 tile per workgroup, 4x4 outputs per invocation.
    Blocked4,
}

impl Variant {
    /// All variants, in benchmark order.
    pub const ALL: [Variant; 3] = [Variant::Naive, Variant::Tiled16, Variant::Blocked4];

    fn source(self) -> &'static str {
        match self {
            Variant::Naive => include_str!("../shaders/matmul/matmul_naive.wgsl"),
            Variant::Tiled16 => include_str!("../shaders/matmul/matmul_tiled16.wgsl"),
            Variant::Blocked4 => include_str!("../shaders/matmul/matmul_blocked4.wgsl"),
        }
    }

    fn key(self) -> &'static str {
        match self {
            Variant::Naive => "matmul_naive",
            Variant::Tiled16 => "matmul_tiled16",
            Variant::Blocked4 => "matmul_blocked4",
        }
    }

    /// Output elements covered by one workgroup, as (columns, rows).
    pub fn tile(self) -> (u32, u32) {
        match self {
            Variant::Naive => (8, 8),
            Variant::Tiled16 => (16, 16),
            Variant::Blocked4 => (64, 64),
        }
    }
}

/// Workgroup counts `(x, y)` to cover an `m x n` output, or an error naming the limit.
pub fn dispatch_size(v: Variant, m: usize, n: usize, max: u32) -> Result<(u32, u32), GpuError> {
    groups_for((n, m), v.tile(), max)
}

/// A matmul kernel bound to a variant; the pipeline is compiled once per device and cached.
pub struct Matmul {
    pipeline: wgpu::ComputePipeline,
    variant: Variant,
}

impl Matmul {
    /// Get (compiling on first use) the variant's pipeline on this device.
    pub fn new(gpu: &Gpu, variant: Variant) -> Result<Matmul, GpuError> {
        let pipeline = gpu.pipeline(variant.key(), variant.source())?;
        Ok(Matmul { pipeline, variant })
    }

    /// Record and submit one multiply on resident buffers (does not wait).
    pub fn submit(
        &self,
        gpu: &Gpu,
        a: &GpuBuf,
        b: &GpuBuf,
        c: &GpuBuf,
        (m, k, n): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(GpuError(format!(
                "matmul buffer sizes {}/{}/{} do not match {m}x{k}x{n}",
                a.len(),
                b.len(),
                c.len()
            )));
        }
        let groups = dispatch_size(self.variant, m, n, gpu.max_workgroups())?;
        gpu.dispatch(
            &self.pipeline,
            &[a, b, c],
            &[m as u32, k as u32, n as u32],
            groups,
        )
    }
}

impl Gpu {
    /// Block until all submitted work has finished.
    pub fn wait(&self) -> Result<(), GpuError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| GpuError(format!("poll failed: {e}")))
    }
}

impl Gpu {
    /// Submit an empty command buffer and wait for it: the fixed cost of one GPU round trip.
    pub fn roundtrip(&self) -> Result<(), GpuError> {
        let enc = self.device.create_command_encoder(&Default::default());
        self.queue.submit([enc.finish()]);
        self.wait()
    }
}

/// Upload, multiply, read back: the host-to-host convenience used by tests.
pub fn matmul_host(
    gpu: &Gpu,
    variant: Variant,
    a: &[f32],
    b: &[f32],
    (m, k, n): (usize, usize, usize),
) -> Result<Vec<f32>, GpuError> {
    let (ab, bb, cb) = (gpu.upload(a)?, gpu.upload(b)?, gpu.zeros(m * n)?);
    Matmul::new(gpu, variant)?.submit(gpu, &ab, &bb, &cb, (m, k, n))?;
    gpu.read(&cb)
}
#[cfg(test)]
mod tests;
