use super::*;

impl Gpu {
    /// A zero-filled buffer of `len` f32 (at least 4 bytes are allocated).
    pub fn zeros(&self, len: usize) -> Result<GpuBuf, GpuError> {
        let bytes = (len.max(1) * 4) as u64;
        if bytes > self.info().max_binding {
            return Err(GpuError(format!(
                "buffer of {bytes} bytes exceeds the adapter binding limit {}",
                self.info().max_binding
            )));
        }
        let buf = self.scoped(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes,
                usage: USAGE,
                mapped_at_creation: false,
            })
        })?;
        self.buffers_created
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(GpuBuf {
            buf,
            len,
            shape: vec![len],
        })
    }

    /// A zero-filled buffer with the given shape.
    pub fn zeros_shaped(&self, shape: &[usize]) -> Result<GpuBuf, GpuError> {
        let mut b = self.zeros(shape.iter().product())?;
        b.shape = shape.to_vec();
        Ok(b)
    }

    /// Upload `data` into a new buffer.
    pub fn upload(&self, data: &[f32]) -> Result<GpuBuf, GpuError> {
        let b = self.zeros(data.len())?;
        if !data.is_empty() {
            self.scoped(|| {
                self.queue
                    .write_buffer(&b.buf, 0, bytemuck::cast_slice(data))
            })?;
        }
        Ok(b)
    }

    /// Overwrite an existing buffer with `data` (same length).
    pub fn copy_from_host(&self, b: &GpuBuf, data: &[f32]) -> Result<(), GpuError> {
        if data.len() != b.len {
            return Err(GpuError(format!(
                "copy_from_host: {} elements into a buffer of {}",
                data.len(),
                b.len
            )));
        }
        if self.batching() {
            return Err(GpuError(
                "copy_from_host while a batch is recording: end the batch first".into(),
            ));
        }
        if !data.is_empty() {
            self.scoped(|| {
                self.queue
                    .write_buffer(&b.buf, 0, bytemuck::cast_slice(data))
            })?;
        }
        Ok(())
    }
}
