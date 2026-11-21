//! Device buffers of f32: upload and readback.
use crate::device::{Gpu, GpuError};
mod gpubuf;
pub use gpubuf::*;
mod part;

#[cfg(test)]
mod tests;
