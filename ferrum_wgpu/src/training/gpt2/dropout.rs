//! FFN-hidden dropout on the GPU, the same masks as the CPU trainer (`CpuTrainer::with_dropout`).
//!
//! After GELU the activations are masked in place (the masked values feed `w2` and are what the
//! backward pass reads), and in the backward pass the gradient of those activations is masked with
//! the same mask before it goes through GELU's derivative. Nothing is stored: the mask is a pure
//! function of (step seed, layer, global unit index), recomputed by the kernel both times.
//!
//! The step seed comes from the run seed and the number of steps taken, as on the CPU, so a
//! resumed run continues the same masks. A micro-batch's units are numbered from the start of the
//! whole batch, so splitting a batch into micro-batches does not change which units drop.

use super::GpuTrainer;
use crate::device::GpuError;
use crate::ops_dropout::{dropout_inplace, Place};
use crate::tensor::GpuBuf;
use ferrum_ref::dropout::{step_seed, Dropout};

impl GpuTrainer<'_> {
    /// Train with FFN-hidden dropout at `rate` (clamped to 0.95; 0 or less turns it off), with
    /// masks derived from `seed` and the step number.
    pub fn set_dropout(&mut self, rate: f32, seed: u64) {
        self.dropout = (rate, seed);
    }

    /// The dropout rate in effect (`0.0` is off).
    pub fn dropout_rate(&self) -> f32 {
        self.dropout.0.clamp(0.0, ferrum_ref::dropout::MAX_RATE)
    }

    /// Number the units of the next recorded forward/backward from global unit `first`.
    pub(super) fn set_dropout_offset(&self, first: usize) -> Result<(), GpuError> {
        let first = u32::try_from(first)
            .map_err(|_| GpuError("dropout: unit indices exceed 32 bits".into()))?;
        self.drop_offset.set(first);
        Ok(())
    }

    /// Record the mask over `x` (a block's FFN activations, or their gradient); a no-op when
    /// dropout is off. A batch must be open.
    pub(super) fn record_dropout(&self, x: &GpuBuf, block: usize) -> Result<(), GpuError> {
        let (rate, seed) = self.dropout;
        let Some(d) = Dropout::new(rate, step_seed(seed, self.t)) else {
            return Ok(());
        };
        let place = Place {
            block: block as u32,
            offset: self.drop_offset.get(),
        };
        dropout_inplace(self.gpu, x, &d, place)
    }
}

#[cfg(test)]
mod tests;
