//! Activations and scratch buffers for one (batch, tokens) shape.
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::gpt2::Config;
mod acts_x;
pub(crate) use acts_x::*;
mod part;
