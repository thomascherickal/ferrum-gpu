//! Device activations and scratch for one `(batch, t)` shape of the Llama trainer.
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::llama::{rope_tables, Config, RopeScaling};
mod buffers;
pub(crate) use buffers::*;
mod acts_alloc;
