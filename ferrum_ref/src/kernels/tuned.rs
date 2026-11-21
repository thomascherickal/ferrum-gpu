// Provenance: algorithm copied from ferrum-cpu `ferrum_core/src/ops.rs` (matmul_block,
// micro_tile, edge_rows; KC/NC cache panels, 4x8 register tile), commit 56a9d3a. Rewritten
// over plain slices with a scoped-thread row split; ours from here on.
//! Tuned CPU matmul: the speed baseline for GPU gates (the simple kernel in [`crate::matmul`]
//! stays the parity oracle). For any output element the contraction sums `k` in ascending
//! order, so the result is bit-identical to [`crate::matmul::matmul`] at any thread count.
use crate::matmul::MatmulError;
mod kc;
use kc::*;
mod edge;
pub use edge::*;

#[cfg(test)]
mod tests;
