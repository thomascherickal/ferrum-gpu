//! Activations and scratch buffers for one (batch, tokens) shape.
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::gpt2::Config;
mod buffers;
pub(crate) use buffers::*;
mod acts_alloc;
