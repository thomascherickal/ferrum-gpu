//! Device activations and scratch for one `(batch, t)` shape of the Llama trainer.
use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::llama::{rope_tables, Config, RopeScaling};
mod acts_x;
pub(crate) use acts_x::*;
mod part;
