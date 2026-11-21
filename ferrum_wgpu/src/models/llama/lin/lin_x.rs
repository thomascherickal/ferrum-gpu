use super::*;

/// A weight matrix `[k, n]` resident on the device.
pub(crate) enum Lin {
    /// Plain f32.
    F32 { w: GpuBuf, k: usize, n: usize },
    /// Block-quantised, dequantised in the kernel.
    Q(QWeight),
    /// Larger than one binding: column blocks, each an ordinary `Lin`, with a zero bias of its own
    /// width for the single-row kernel. Each block's output is copied to its columns of `y`.
    // ponytail: scratch for the block outputs is allocated on every call; cache it if a sharded
    // model is ever decoded fast enough for the allocations to show up.
    Sharded {
        parts: Vec<(usize, usize, Lin, GpuBuf)>,
        n: usize,
    },
}
