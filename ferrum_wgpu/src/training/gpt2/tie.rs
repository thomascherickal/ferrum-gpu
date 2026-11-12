//! Tied embedding and output head on the GPU: one shared matrix `W`, with `tok = W` and
//! `head = Wᵀ`, the same scheme as the CPU trainer (`CpuTrainer::with_tie`).
//!
//! `dL/dW = d_tok + d_headᵀ`, so the head's gradient is added into the embedding's before the
//! optimizer, and the head takes no part in the norm or in AdamW (its gradient is "zero" and it is
//! rewritten from the embedding after every update). The head is also re-derived whenever the
//! weights change under it: when tying is switched on, after `load`, and after the QAT snap.

use super::GpuTrainer;
use crate::device::GpuError;
use crate::ops_opt::{parts_for, transpose_into};
use crate::tensor::GpuBuf;

impl GpuTrainer<'_> {
    /// Canonical index of the output head: after the position table, the blocks, and `lnf`.
    pub(super) fn head_index(&self) -> usize {
        2 + 16 * self.cfg.layers + 2
    }

    /// Whether the head is tied to the embedding.
    pub fn tie_enabled(&self) -> bool {
        self.tie
    }

    /// Tie (or untie) the output head to the token embedding. Tying rewrites the head as the
    /// transposed embedding at once.
    pub fn set_tie(&mut self, on: bool) -> Result<(), GpuError> {
        if !on {
            self.tie = false;
            return Ok(());
        }
        self.tie = true;
        // the head no longer contributes to the gradient norm: clear its partial sums once, so
        // sums left from untied steps cannot leak into the clip scale
        let (i, n) = (
            self.head_index(),
            parts_for(self.params[self.head_index()].len()),
        );
        let zeros = self.gpu.zeros(n)?;
        self.gpu.begin_batch()?;
        let rec = (|| {
            self.gpu
                .copy_f32(&zeros, 0, &self.part, self.part_offsets[i], n)?;
            self.record_derive_head(&self.params)
        })();
        self.gpu.end_batch(rec.is_ok())?;
        rec
    }

    /// Record `head = tokᵀ` in `set` (a batch must be open).
    pub(super) fn record_derive_head(&self, set: &[GpuBuf]) -> Result<(), GpuError> {
        let (v, d) = (self.cfg.vocab, self.cfg.dim);
        transpose_into(self.gpu, &set[0], &set[self.head_index()], (d, v), false)
    }

    /// Record `d_tok += d_headᵀ` in the gradients `g` (a batch must be open).
    pub(super) fn record_fold_head_grad(&self, g: &[GpuBuf]) -> Result<(), GpuError> {
        let (v, d) = (self.cfg.vocab, self.cfg.dim);
        transpose_into(self.gpu, &g[self.head_index()], &g[0], (v, d), true)
    }

    /// Re-derive the head outside a recorded step (after the weights were replaced).
    pub(super) fn derive_head_now(&self) -> Result<(), GpuError> {
        self.gpu.begin_batch()?;
        let rec = self.record_derive_head(&self.params);
        self.gpu.end_batch(rec.is_ok())?;
        rec
    }
}

#[cfg(test)]
mod tests;
