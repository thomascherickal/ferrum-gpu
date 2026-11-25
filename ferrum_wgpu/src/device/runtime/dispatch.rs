use super::*;

impl Gpu {
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
    pub(crate) fn record(&self, f: impl FnOnce(&mut wgpu::CommandEncoder)) -> Result<(), GpuError> {
        self.check()?;
        #[cfg(any(test, feature = "fault-injection"))]
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

    /// Record `commands` into one batch: submitted if they all succeed, dropped if one fails
    /// (that error is returned).
    pub fn batched(&self, commands: impl FnOnce() -> Result<(), GpuError>) -> Result<(), GpuError> {
        self.begin_batch()?;
        let rec = commands();
        self.end_batch(rec.is_ok())?;
        rec
    }
}
