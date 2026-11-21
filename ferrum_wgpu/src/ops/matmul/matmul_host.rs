use super::*;

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
