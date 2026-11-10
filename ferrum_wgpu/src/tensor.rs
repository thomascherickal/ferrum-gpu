//! Device buffers of f32: upload and readback.

use crate::device::{Gpu, GpuError};

/// A device buffer holding `len` f32 values with a row-major shape (`[len]` unless set).
#[derive(Debug)]
pub struct GpuBuf {
    pub(crate) buf: wgpu::Buffer,
    len: usize,
    shape: Vec<usize>,
}

impl GpuBuf {
    /// Number of f32 elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The row-major shape; its product is `len`.
    pub fn shape(&self) -> &[usize] {
        &self.shape
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
    fn shapes_are_kept_and_checked() {
        let Some(g) = open_for_test() else { return };
        let b = g
            .upload_shaped(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.len(), 6);
        assert_eq!(g.zeros(4).unwrap().shape(), &[4]);
        assert_eq!(g.zeros_shaped(&[3, 0]).unwrap().len(), 0);
        let e = g
            .upload_shaped(&[1.0, 2.0], &[2, 2])
            .expect_err("must refuse");
        assert!(e.to_string().contains("[2, 2]") && e.to_string().contains("2 were given"));
    }

    #[test]
    fn copy_from_host_overwrites_in_place_and_checks_length() {
        let Some(g) = open_for_test() else { return };
        let b = g.upload(&[1.0, 2.0, 3.0]).unwrap();
        g.copy_from_host(&b, &[7.0, 8.0, 9.0]).unwrap();
        assert_eq!(g.read(&b).unwrap(), vec![7.0, 8.0, 9.0]);
        g.copy_from_host(&g.zeros(0).unwrap(), &[]).unwrap();
        assert!(g
            .copy_from_host(&b, &[1.0])
            .unwrap_err()
            .to_string()
            .contains("1 elements into a buffer of 3"));
        g.begin_batch().unwrap();
        assert!(g
            .copy_from_host(&b, &[1.0, 1.0, 1.0])
            .unwrap_err()
            .to_string()
            .contains("batch"));
        g.end_batch(false).unwrap();
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
        let e = g.zeros(too_many).expect_err("must refuse");
        assert!(e.to_string().contains("binding limit"));
    }
}
