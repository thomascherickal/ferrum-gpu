//! Dropout masks for training (ferrum-cpu's scheme: on the FFN hidden activations, after GELU).
//!
//! Inverted dropout: each unit is zeroed with probability `rate` and the survivors are scaled by
//! `1 / (1 - rate)`, so the expected activation is unchanged and inference needs no correction.
//!
//! A mask is a pure function of `(seed, block, unit index)`: a counter-based 32-bit hash, not a
//! stream. So the same step gives the same mask however the work is split (threads, batches,
//! micro-batches), a resumed run reproduces the masks of the run it continues, and the GPU
//! computes any unit's mask on its own with the same integer arithmetic (WGSL `u32` wraps like
//! Rust's `wrapping_*`; the final `u32 >> 8` to `f32` is exact), so CPU and GPU masks are
//! bit-identical.

#[cfg(test)]
mod grad_tests;
#[cfg(test)]
mod hash_tests;
mod mask;
pub use mask::*;

#[cfg(test)]
mod tests;
