//! Quantisation-aware training for the GPU trainer: snapped copies of the weights feed the
//! forward and backward pass, while Adam keeps updating the full-precision masters
//! (a straight-through estimator). The snap is bit-exact with the CPU reference and with the
//! Q8_0 encoder the GGUF export uses.

use super::GpuTrainer;
use crate::budget::BufferPlan;
use crate::device::GpuError;
use crate::ops_qat::{q8_snap_cols, q8_snap_rows};
use crate::tensor::GpuBuf;
use ferrum_ref::gpt2::Config;

/// How the canonical parameter `i` of a model is treated by the snap.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Snap {
    /// Token embedding, blocks of 32 along each row (row length `dim`).
    Rows(usize),
    /// An `[in, out]` matrix, blocks down each column.
    Cols(usize, usize),
    /// Stored F32 in a Q8_0 export (position embeddings, biases, LayerNorm): copied unchanged.
    Keep,
}

/// What the snap does to canonical parameter `i`, mirroring which tensors the GGUF writer stores
/// as Q8_0.
pub(super) fn snap_kind(c: &Config, i: usize) -> Snap {
    let (d, h, v) = (c.dim, c.hidden, c.vocab);
    let blocks_end = 2 + 16 * c.layers;
    match i {
        0 => Snap::Rows(d),
        1 => Snap::Keep,
        _ if i < blocks_end => match (i - 2) % 16 {
            2 | 4 | 6 | 8 => Snap::Cols(d, d),
            12 => Snap::Cols(d, h),
            14 => Snap::Cols(h, d),
            _ => Snap::Keep,
        },
        _ if i == blocks_end + 2 => Snap::Cols(d, v),
        _ => Snap::Keep,
    }
}

/// The snapped copies QAT keeps: one buffer per parameter (unsnapped ones are plain copies, so
/// the forward pass can index one list).
pub fn qat_plan(c: &Config) -> BufferPlan {
    let mut p = BufferPlan::default();
    for (i, len) in ferrum_ref::gpt2::param_lens(c)
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        p.add(&format!("fwd{i}"), (len * 4) as u64);
    }
    p
}

impl GpuTrainer<'_> {
    /// Record the snap of every parameter into the forward-weights buffers (a batch must be open).
    pub(super) fn record_snap(&self) -> Result<(), GpuError> {
        let fwd = self.fwd.as_ref().expect("qat buffers allocated");
        for (i, (src, dst)) in self.params.iter().zip(fwd).enumerate() {
            match snap_kind(&self.cfg, i) {
                Snap::Rows(ne0) => q8_snap_rows(self.gpu, src, dst, ne0)?,
                Snap::Cols(r, c) => q8_snap_cols(self.gpu, src, dst, (r, c))?,
                Snap::Keep => self.gpu.copy_f32(src, 0, dst, 0, src.len())?,
            }
        }
        if self.tie {
            self.record_derive_head(fwd)?; // the head the forward pass sees is the snapped embedding's
        }
        Ok(())
    }

    /// The weights the forward and backward pass read: the snapped copies under QAT, else the masters.
    pub(super) fn forward_weights(&self) -> &Vec<GpuBuf> {
        match self.fwd.as_ref() {
            Some(f) if self.qat => f,
            _ => &self.params,
        }
    }

    /// Turn quantisation-aware training on or off. Enabling allocates one more copy of the
    /// parameters (checked against the budget first); disabling frees it.
    pub fn set_qat(&mut self, on: bool) -> Result<(), GpuError> {
        if !on {
            self.qat = false;
            self.fwd = None;
            return Ok(());
        }
        if self.fwd.is_none() {
            let mut plan = super::plan::static_plan(&self.cfg);
            qat_plan(&self.cfg)
                .rows()
                .iter()
                .for_each(|(n, b)| plan.add(n, *b));
            plan.check(self.gpu.info().max_binding, self.budget)
                .map_err(|e| GpuError(format!("quantisation-aware training does not fit: {e}")))?;
            let bufs = self
                .params
                .iter()
                .map(|p| self.gpu.zeros(p.len()))
                .collect::<Result<Vec<_>, _>>()?;
            self.fwd = Some(bufs);
        }
        self.qat = true;
        Ok(())
    }

    /// Whether quantisation-aware training is on.
    pub fn qat_enabled(&self) -> bool {
        self.qat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            vocab: 16,
            ctx: 8,
            dim: 32,
            heads: 2,
            layers: 2,
            hidden: 64,
        }
    }

    #[test]
    fn the_snap_plan_matches_the_tensors_a_q8_0_export_quantises() {
        let c = cfg();
        // Count which canonical parameters are quantised and compare with the CPU reference,
        // which snaps exactly the tensors the GGUF writer stores as Q8_0.
        let w = ferrum_ref::gpt2::Gpt2::random(c, 3);
        let snapped = w.fake_quantize();
        for (i, (a, b)) in w.params().iter().zip(snapped.params()).enumerate() {
            let changed = a != &b;
            let planned = snap_kind(&c, i) != Snap::Keep;
            let kind = snap_kind(&c, i);
            assert_eq!(changed, planned, "parameter {i}: kind {kind:?}");
        }
        assert_eq!(snap_kind(&c, 0), Snap::Rows(32));
        assert_eq!(snap_kind(&c, 2 + 12), Snap::Cols(32, 64), "w1");
        assert_eq!(snap_kind(&c, 2 + 14), Snap::Cols(64, 32), "w2");
        assert_eq!(
            snap_kind(&c, 2 + 16 + 2),
            Snap::Cols(32, 32),
            "second block's wq"
        );
        assert_eq!(snap_kind(&c, 2 + 32 + 2), Snap::Cols(32, 16), "head");
    }

    #[test]
    fn the_plan_has_one_buffer_per_parameter() {
        let c = cfg();
        let p = qat_plan(&c);
        assert_eq!(p.rows().len(), 2 + 16 * c.layers + 3);
        let total: u64 = ferrum_ref::gpt2::Gpt2::random(c, 1).num_params() as u64 * 4;
        assert_eq!(p.total(), total);
        // An invalid config yields an empty plan rather than panicking.
        assert!(qat_plan(&Config { heads: 3, ..c }).rows().is_empty());
    }
}
