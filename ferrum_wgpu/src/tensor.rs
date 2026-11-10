//! Device buffers of f32: upload and readback.

use crate::device::{Gpu, GpuError};

/// A device buffer holding `len` f32 values.
pub struct GpuBuf {
    pub(crate) buf: wgpu::Buffer,
    len: usize,
}

impl GpuBuf {
    /// Number of f32 elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

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
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: USAGE,
            mapped_at_creation: false,
        });
        Ok(GpuBuf { buf, len })
    }

    /// Upload `data` into a new buffer.
    pub fn upload(&self, data: &[f32]) -> Result<GpuBuf, GpuError> {
        let b = self.zeros(data.len())?;
        if !data.is_empty() {
            self.queue
                .write_buffer(&b.buf, 0, bytemuck::cast_slice(data));
        }
        Ok(b)
    }

    /// Copy the buffer back to the host (the single readback path).
    pub fn read(&self, b: &GpuBuf) -> Result<Vec<f32>, GpuError> {
        if b.len == 0 {
            return Ok(Vec::new());
        }
        let bytes = (b.len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&b.buf, 0, &staging, 0, bytes);
        self.queue.submit([enc.finish()]);
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
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::device::open_for_test;

    #[test]
    fn round_trip_odd_lengths_and_empty() {
        let Some(g) = open_for_test() else { return };
        for n in [0usize, 1, 3, 7, 1001] {
            let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
            let b = g.upload(&data).unwrap();
            assert_eq!(b.len(), n);
            assert_eq!(b.is_empty(), n == 0);
            assert_eq!(g.read(&b).unwrap(), data, "len {n}");
        }
    }

    #[test]
    fn zeros_are_zero() {
        let Some(g) = open_for_test() else { return };
        assert_eq!(g.read(&g.zeros(5).unwrap()).unwrap(), vec![0.0; 5]);
    }

    #[test]
    fn a_buffer_over_the_binding_limit_is_refused_by_name_of_limit() {
        let Some(g) = open_for_test() else { return };
        let too_many = (g.info().max_binding / 4 + 1) as usize;
        let e = g.zeros(too_many).err().expect("must refuse");
        assert!(e.to_string().contains("binding limit"));
    }
}
