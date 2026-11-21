//! The forward and backward dispatch sequence of one training step.
use super::GpuTrainer;
use crate::device::GpuError;
use crate::ops::{self, add};
use crate::ops_bwd as bwd;
use crate::tensor::GpuBuf;
mod pi;
use pi::*;
mod part;
