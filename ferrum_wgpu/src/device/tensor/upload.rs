use super::*;

impl Gpu {
    /// Upload raw 4-byte words (`words` of them) as a buffer, e.g. u32 indices.
    pub fn upload_raw(&self, bytes: &[u8], words: usize) -> Result<GpuBuf, GpuError> {
        debug_assert_eq!(bytes.len(), words * 4);
        let b = self.zeros(words)?;
        if words > 0 {
            self.scoped(|| self.queue.write_buffer(&b.buf, 0, bytes))?;
        }
        Ok(b)
    }

    /// Upload `data` with a shape; the shape's product must equal `data.len()`.
    pub fn upload_shaped(&self, data: &[f32], shape: &[usize]) -> Result<GpuBuf, GpuError> {
        let want: usize = shape.iter().product();
        if want != data.len() {
            return Err(GpuError(format!(
                "shape {shape:?} holds {want} elements but {} were given",
                data.len()
            )));
        }
        let mut b = self.upload(data)?;
        b.shape = shape.to_vec();
        Ok(b)
    }

    /// Copy the buffer back to the host (the single readback path).
    pub fn read(&self, b: &GpuBuf) -> Result<Vec<f32>, GpuError> {
        if self.batching() {
            return Err(GpuError(
                "read while a batch is recording: end the batch first".into(),
            ));
        }
        if b.len == 0 {
            return Ok(Vec::new());
        }
        let bytes = (b.len * 4) as u64;
        let staging = self.scoped(|| {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&b.buf, 0, &staging, 0, bytes);
            self.queue.submit([enc.finish()]);
            staging
        })?;
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError(format!("poll failed: {e}")))?;
        rx.recv()
            .map_err(|e| GpuError(format!("map callback lost: {e}")))?
            .map_err(|e| GpuError(format!("map failed: {e}")))?;
        let view = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| GpuError(format!("mapped range: {e}")))?;
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        staging.unmap();
        self.check()?;
        Ok(out)
    }
}
