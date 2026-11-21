//! Device buffers of f32: upload and readback.
use crate::device::{Gpu, GpuError};
mod gpubuf;
pub use gpubuf::*;
mod alloc;
mod upload;

#[cfg(test)]
mod tests;
