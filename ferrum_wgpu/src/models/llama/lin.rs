//! One linear layer on the device: f32 or packed-quantised weights, applied to one row (decode) or
//! many (prefill, scoring), with an optional bias.
use super::shard::{blocks, column_block};
use crate::device::{Gpu, GpuError};
use crate::matmul::Matmul;
use crate::ops;
use crate::ops_quant::{effective, linear_q, packed_words, upload_quantised, QKind, QWeight};
use crate::tensor::GpuBuf;
mod lin_x;
pub(crate) use lin_x::*;
mod part;
