//! Cached single-token decoding on the CPU (the speed baseline and the cache oracle).
use super::Gpt2;
use crate::nn;
mod cache;
pub use cache::*;
mod feed;
