//! Matmul kernels `C[m,n] = A[m,k] * B[k,n]` on the GPU: three variants for the Phase 0 spike.
use crate::device::{Gpu, GpuError};
use crate::runtime::groups_for;
use crate::tensor::GpuBuf;
mod variant;
pub use variant::*;
mod matmul_host;
pub use matmul_host::*;

#[cfg(test)]
mod tests;
