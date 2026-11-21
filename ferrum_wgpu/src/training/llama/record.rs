//! The forward and backward dispatch sequence of one Llama training step.
use super::GpuLlamaTrainer;
use crate::device::{Gpu, GpuError};
use crate::ops::{self, add};
use crate::ops_bwd::{colsum, embed_bwd, matmul_ex};
use crate::ops_llama::{attention_gqa, gather_rows, rmsnorm, rope, swiglu};
use crate::ops_llama_bwd::{attention_gqa_bwd, rmsnorm_bwd, swiglu_bwd};
use crate::tensor::GpuBuf;
use ferrum_ref::llama::BLOCK_PARAMS;
mod part;
mod pi_m;
use pi_m::*;
