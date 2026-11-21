/// A device buffer holding `len` f32 values with a row-major shape (`[len]` unless set).
#[derive(Debug)]
pub struct GpuBuf {
    pub(crate) buf: wgpu::Buffer,
    pub(super) len: usize,
    pub(super) shape: Vec<usize>,
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

pub(super) const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);
