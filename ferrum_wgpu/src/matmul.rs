//! Matmul kernels `C[m,n] = A[m,k] * B[k,n]` on the GPU: three variants for the Phase 0 spike.

use crate::device::{Gpu, GpuError};
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
            Variant::Naive => include_str!("shaders/matmul_naive.wgsl"),
            Variant::Tiled16 => include_str!("shaders/matmul_tiled16.wgsl"),
            Variant::Blocked4 => include_str!("shaders/matmul_blocked4.wgsl"),
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
    let (tc, tr) = v.tile();
    let x = n.div_ceil(tc as usize);
    let y = m.div_ceil(tr as usize);
    if x > max as usize || y > max as usize {
        return Err(GpuError(format!(
            "matmul {m}x{n} needs {x}x{y} workgroups, adapter allows {max} per dimension"
        )));
    }
    Ok((x as u32, y as u32))
}

/// A compiled matmul pipeline.
pub struct Matmul {
    pipeline: wgpu::ComputePipeline,
    variant: Variant,
}

impl Matmul {
    /// Compile the variant on this device.
    pub fn new(gpu: &Gpu, variant: Variant) -> Matmul {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul"),
                source: wgpu::ShaderSource::Wgsl(variant.source().into()),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Matmul { pipeline, variant }
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
        let (gx, gy) = dispatch_size(self.variant, m, n, gpu.max_workgroups())?;
        let dims = [m as u32, k as u32, n as u32, 0u32];
        let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&ubuf, 0, bytemuck::cast_slice(&dims));
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ubuf.as_entire_binding(),
                },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        gpu.queue.submit([enc.finish()]);
        Ok(())
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

/// Upload, multiply, read back: the host-to-host convenience used by tests.
pub fn matmul_host(
    gpu: &Gpu,
    variant: Variant,
    a: &[f32],
    b: &[f32],
    (m, k, n): (usize, usize, usize),
) -> Result<Vec<f32>, GpuError> {
    let (ab, bb, cb) = (gpu.upload(a)?, gpu.upload(b)?, gpu.zeros(m * n)?);
    Matmul::new(gpu, variant).submit(gpu, &ab, &bb, &cb, (m, k, n))?;
    gpu.read(&cb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::open_for_test;
    use ferrum_ref::matmul::{matmul, rand_vec};

    fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
            .fold(0.0, f32::max)
    }

    #[test]
    fn dispatch_size_covers_the_output_and_enforces_the_limit() {
        assert_eq!(dispatch_size(Variant::Naive, 9, 17, 65535).unwrap(), (3, 2));
        assert_eq!(
            dispatch_size(Variant::Tiled16, 16, 16, 65535).unwrap(),
            (1, 1)
        );
        assert_eq!(
            dispatch_size(Variant::Blocked4, 65, 1, 65535).unwrap(),
            (1, 2)
        );
        let e = dispatch_size(Variant::Naive, 8 * 70_000, 8, 65535).unwrap_err();
        assert!(e.to_string().contains("65535"));
    }

    #[test]
    fn every_variant_matches_the_cpu_reference_on_odd_and_degenerate_shapes() {
        let Some(g) = open_for_test() else { return };
        let shapes = [
            (1, 1, 1),
            (1, 7, 5),
            (5, 1, 9),
            (17, 33, 15),
            (64, 64, 64),
            (65, 17, 63),
            (3, 129, 130),
            (100, 3, 1),
        ];
        for v in Variant::ALL {
            for &(m, k, n) in &shapes {
                let a = rand_vec(m * k, 11);
                let b = rand_vec(k * n, 12);
                let want = matmul(&a, &b, m, k, n).unwrap();
                let got = matmul_host(&g, v, &a, &b, (m, k, n)).unwrap();
                let err = max_rel_err(&got, &want);
                assert!(err < 1e-4, "{v:?} {m}x{k}x{n}: max rel err {err}");
            }
        }
    }

    #[test]
    fn repeat_runs_are_bit_identical() {
        let Some(g) = open_for_test() else { return };
        let (m, k, n) = (70, 50, 90);
        let a = rand_vec(m * k, 5);
        let b = rand_vec(k * n, 6);
        for v in Variant::ALL {
            let first = matmul_host(&g, v, &a, &b, (m, k, n)).unwrap();
            let second = matmul_host(&g, v, &a, &b, (m, k, n)).unwrap();
            assert_eq!(first, second, "{v:?}");
        }
    }

    #[test]
    fn mismatched_buffers_are_rejected() {
        let Some(g) = open_for_test() else { return };
        let mm = Matmul::new(&g, Variant::Naive);
        let (a, b, c) = (
            g.zeros(4).unwrap(),
            g.zeros(4).unwrap(),
            g.zeros(3).unwrap(),
        );
        let e = mm.submit(&g, &a, &b, &c, (2, 2, 2)).unwrap_err();
        assert!(e.to_string().contains("do not match"));
    }
}
