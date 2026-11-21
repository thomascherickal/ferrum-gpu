use super::*;

/// Runs training and decoding on an adapter.
pub struct GpuBackend {
    pub(super) gpu: Arc<Gpu>,
    /// How decoder weights are stored on the device (`None` is f32). Training is always f32.
    pub(super) weights: Option<QKind>,
}

impl GpuBackend {
    /// Wrap an opened adapter.
    pub fn new(gpu: Arc<Gpu>) -> Self {
        GpuBackend { gpu, weights: None }
    }

    /// Store decoder weights in `weights` on the device (converted at load). Matrices whose input
    /// width is not a whole number of 32-blocks stay f32.
    pub fn with_weights(mut self, weights: Option<QKind>) -> Self {
        self.weights = weights;
        self
    }

    /// Open the named adapter (or the best hardware one).
    pub fn open(name: Option<&str>) -> Result<Self, String> {
        Gpu::open(name)
            .map(|g| GpuBackend::new(Arc::new(g)))
            .map_err(|e| e.to_string())
    }
}
