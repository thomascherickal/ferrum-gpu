// Provenance: logic copied from ferrum-cpu `ferrum_core/src/ops.rs` (`matmul`,
// i-k-j loop order), commit 56a9d3a. Rewritten over plain slices; ours from here on.
//! Row-major f32 matrix multiply `[m,k] x [k,n] -> [m,n]`.

mod matmulerror;
pub use matmulerror::*;
mod matmul_threaded;
pub use matmul_threaded::*;

#[cfg(test)]
mod tests;
