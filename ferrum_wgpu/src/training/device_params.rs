//! What both trainers keep per parameter on the device: weights, gradients, AdamW moments, and
//! the gradient-norm scratch; plus the AdamW step over them.

use crate::device::{Gpu, GpuError};
use crate::ops_opt::{adam_update, norm_scale, parts_for, sumsq_parts};
use crate::tensor::GpuBuf;
use ferrum_ref::trainer::StepConfig;

/// Per-parameter device state of a model being trained, in the model's `params()` order.
pub(crate) struct DeviceParams {
    pub(crate) params: Vec<GpuBuf>,
    pub(crate) grads: Vec<GpuBuf>,
    pub(crate) m: Vec<GpuBuf>,
    pub(crate) v: Vec<GpuBuf>,
    /// Whether each parameter takes weight decay.
    pub(crate) decays: Vec<bool>,
    /// Partial sums of squares of every gradient (see `parts_for`), at `part_offsets[i]`.
    pub(crate) part: GpuBuf,
    pub(crate) part_offsets: Vec<usize>,
    /// `[clip scale, pre-clip norm]`.
    pub(crate) scale: GpuBuf,
}

impl DeviceParams {
    /// Upload `weights` with zeroed gradients and moments; `decay(i)` says whether parameter `i`
    /// takes weight decay.
    pub(crate) fn upload(
        gpu: &Gpu,
        weights: &[&Vec<f32>],
        decay: impl Fn(usize) -> bool,
    ) -> Result<Self, GpuError> {
        let zeros = || -> Result<Vec<GpuBuf>, GpuError> {
            weights.iter().map(|w| gpu.zeros(w.len())).collect()
        };
        let mut part_offsets = Vec::with_capacity(weights.len());
        let mut total = 0;
        for w in weights {
            part_offsets.push(total);
            total += parts_for(w.len());
        }
        Ok(DeviceParams {
            params: weights
                .iter()
                .map(|w| gpu.upload(w))
                .collect::<Result<_, _>>()?,
            grads: zeros()?,
            m: zeros()?,
            v: zeros()?,
            decays: (0..weights.len()).map(decay).collect(),
            part: gpu.zeros(total)?,
            part_offsets,
            scale: gpu.zeros(2)?,
        })
    }

    /// Record the gradient norm, clip scale and AdamW over the gradients in `source` whose index
    /// passes `active` (the commands only; a batch is normally open).
    pub(crate) fn optimize(
        &self,
        gpu: &Gpu,
        source: &[GpuBuf],
        active: impl Fn(usize, &GpuBuf) -> bool,
        step: u64,
        cfg: &StepConfig,
    ) -> Result<(), GpuError> {
        let used = || source.iter().enumerate().filter(|(i, g)| active(*i, g));
        for (i, g) in used() {
            sumsq_parts(gpu, g, &self.part, self.part_offsets[i])?;
        }
        norm_scale(gpu, &self.part, &self.scale, self.part.len(), cfg.clip)?;
        for (i, g) in used() {
            let state = (&self.params[i], g, &self.m[i], &self.v[i]);
            adam_update(gpu, state, &self.scale, &cfg.adam, self.decays[i], step)?;
        }
        Ok(())
    }
}
