use super::*;

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

    pub(super) fn source(self) -> &'static str {
        match self {
            Variant::Naive => include_str!("../../shaders/matmul/matmul_naive.wgsl"),
            Variant::Tiled16 => include_str!("../../shaders/matmul/matmul_tiled16.wgsl"),
            Variant::Blocked4 => include_str!("../../shaders/matmul/matmul_blocked4.wgsl"),
        }
    }

    pub(super) fn key(self) -> &'static str {
        match self {
            Variant::Naive => "matmul_naive",
            Variant::Tiled16 => "matmul_tiled16",
            Variant::Blocked4 => "matmul_blocked4",
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
    groups_for((n, m), v.tile(), max)
}

/// A matmul kernel bound to a variant; the pipeline is compiled once per device and cached.
pub struct Matmul {
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) variant: Variant,
}
