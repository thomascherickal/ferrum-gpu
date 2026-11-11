//! Llama / Qwen2 fine-tuning on the GPU (f32 weights, AdamW). Weights, gradients, Adam moments
//! and every activation stay resident; one step is recorded as a single command batch and only
//! the loss and the gradient norm come back to the host. The CPU oracle is
//! `ferrum_ref::llama::LlamaTrainer`.
//! ponytail: no micro-batching, QAT, weight tying or dropout, and one batch shape's activations
//! are kept at a time (reallocated when `(batch, t)` changes); the shared `Engine` trait and the
//! training loop are still GPT-2 shaped (`Snapshot` holds a `Gpt2`). Add them when a run needs
//! them.

mod acts;
mod plan;
mod record;
#[cfg(test)]
mod tests;

use crate::budget::Budget;
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::ops_opt::{adam_update, norm_scale, parts_for, sumsq_parts};
use crate::tensor::GpuBuf;
use acts::Acts;
use ferrum_ref::llama::Llama;
use ferrum_ref::trainer::{StepConfig, StepStats};

pub use plan::{activation_plan, param_sizes, static_plan, static_plan_for};

fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

/// Everything needed to continue a run: weights, Adam moments, steps taken.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaSnapshot {
    /// Weights.
    pub weights: Llama,
    /// First moments.
    pub m: Llama,
    /// Second moments.
    pub v: Llama,
    /// Optimizer steps completed.
    pub steps: u64,
}

/// A Llama-family model being fine-tuned on the GPU.
pub struct GpuLlamaTrainer<'g> {
    gpu: &'g Gpu,
    cfg: ferrum_ref::llama::Config,
    /// The model's shape with every parameter emptied (read-backs fill it).
    template: Llama,
    budget: Option<Budget>,
    params: Vec<GpuBuf>,
    grads: Vec<GpuBuf>,
    m: Vec<GpuBuf>,
    v: Vec<GpuBuf>,
    decays: Vec<bool>,
    part: GpuBuf,
    part_offsets: Vec<usize>,
    scale: GpuBuf,
    matmul: Matmul,
    acts: Option<Acts>,
    t: u64,
}

impl<'g> GpuLlamaTrainer<'g> {
    /// Upload `w`. Weights, gradients and moments are planned and checked against the adapter
    /// and `budget` before anything is allocated; activations are checked (together with the
    /// static part) when a batch shape is first seen. Tied heads are refused.
    pub fn new(gpu: &'g Gpu, w: &Llama, budget: Option<Budget>) -> Result<Self, GpuError> {
        w.cfg.validate().map_err(err)?;
        w.rope.check(w.cfg.rope_dim).map_err(err)?;
        if w.tied {
            return Err(err(
                "training a tied head is not supported: train an untied copy",
            ));
        }
        static_plan(w)
            .check(gpu.info().max_binding, budget)
            .map_err(|e| err(e.to_string()))?;
        let up = |m: &Llama| {
            m.params()
                .iter()
                .map(|p| gpu.upload(p))
                .collect::<Result<Vec<_>, _>>()
        };
        let zeros = w.zeros_like();
        let sizes: Vec<usize> = w.params().iter().map(|p| p.len()).collect();
        let mut part_offsets = Vec::with_capacity(sizes.len());
        let mut total = 0;
        for &n in &sizes {
            part_offsets.push(total);
            total += parts_for(n);
        }
        let mut template = w.clone();
        template
            .params_mut()
            .into_iter()
            .for_each(|p| *p = Vec::new());
        Ok(GpuLlamaTrainer {
            gpu,
            cfg: w.cfg,
            template,
            budget,
            params: up(w)?,
            grads: up(&zeros)?,
            m: up(&zeros)?,
            v: up(&zeros)?,
            decays: (0..sizes.len()).map(|i| w.decays(i)).collect(),
            part: gpu.zeros(total)?,
            part_offsets,
            scale: gpu.zeros(2)?,
            matmul: Matmul::new(gpu, Variant::Blocked4)?,
            acts: None,
            t: 0,
        })
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &ferrum_ref::llama::Config {
        &self.cfg
    }

    /// Optimizer steps taken.
    pub fn steps(&self) -> u64 {
        self.t
    }

    fn host(&self, bufs: &[GpuBuf]) -> Result<Llama, GpuError> {
        let mut out = self.template.clone();
        for (dst, b) in out.params_mut().into_iter().zip(bufs) {
            *dst = if b.is_empty() {
                Vec::new()
            } else {
                self.gpu.read(b)?
            };
        }
        Ok(out)
    }

    /// Copy the weights back to the host.
    pub fn weights_host(&self) -> Result<Llama, GpuError> {
        self.host(&self.params)
    }

    /// Copy the gradients of the last forward/backward back to the host.
    pub fn grads_host(&self) -> Result<Llama, GpuError> {
        self.host(&self.grads)
    }

    /// Copy the Adam moments back to the host as `(m, v)`.
    pub fn moments_host(&self) -> Result<(Llama, Llama), GpuError> {
        Ok((self.host(&self.m)?, self.host(&self.v)?))
    }

    /// Copy the full training state to the host.
    pub fn snapshot(&self) -> Result<LlamaSnapshot, GpuError> {
        let (m, v) = self.moments_host()?;
        Ok(LlamaSnapshot {
            weights: self.weights_host()?,
            m,
            v,
            steps: self.t,
        })
    }

    /// Replace weights, Adam moments and the step counter (resume from a snapshot).
    pub fn restore(&mut self, s: &LlamaSnapshot) -> Result<(), GpuError> {
        let shape = |m: &Llama| {
            m.cfg == self.cfg
                && m.params()
                    .iter()
                    .zip(&self.params)
                    .all(|(p, b)| p.len() == b.len())
        };
        if !(shape(&s.weights) && shape(&s.m) && shape(&s.v)) {
            return Err(err("restore: the model shape differs from the trainer's"));
        }
        for (set, src) in [(&self.params, &s.weights), (&self.m, &s.m), (&self.v, &s.v)] {
            for (b, p) in set.iter().zip(src.params()) {
                if !p.is_empty() {
                    self.gpu.copy_from_host(b, p)?;
                }
            }
        }
        self.t = s.steps;
        Ok(())
    }

    fn validate(&self, ids: &[usize], targets: &[usize], batch: usize) -> Result<usize, GpuError> {
        let c = &self.cfg;
        if batch == 0 || ids.is_empty() || !ids.len().is_multiple_of(batch) {
            return Err(err(format!(
                "{} tokens do not split into {batch} sequences",
                ids.len()
            )));
        }
        let t = ids.len() / batch;
        if t > c.ctx {
            return Err(err(format!(
                "sequence length {t} exceeds the context {}",
                c.ctx
            )));
        }
        if targets.len() != ids.len() {
            return Err(err(format!(
                "{} targets for {} tokens",
                targets.len(),
                ids.len()
            )));
        }
        if let Some(bad) = ids.iter().chain(targets).find(|&&i| i >= c.vocab) {
            return Err(err(format!(
                "token {bad} is out of range for vocab {}",
                c.vocab
            )));
        }
        Ok(t)
    }

    /// Make sure activations for `(batch, t)` exist, checking the whole footprint first.
    fn ensure_acts(&mut self, batch: usize, t: usize) -> Result<(), GpuError> {
        if self
            .acts
            .as_ref()
            .is_some_and(|a| a.batch == batch && a.t == t)
        {
            return Ok(());
        }
        self.acts = None; // free the old shape before allocating the new one
        let mut plan = static_plan(&self.template_with_sizes());
        activation_plan(&self.cfg, batch, t)
            .rows()
            .iter()
            .for_each(|(n, b)| plan.add(n, *b));
        plan.check(self.gpu.info().max_binding, self.budget)
            .map_err(|e| err(format!("batch {batch} x {t} tokens does not fit: {e}")))?;
        let rope = self.template.rope.clone();
        self.acts = Some(Acts::new(self.gpu, &self.cfg, &rope, batch, t)?);
        Ok(())
    }

    /// The template with the parameter sizes filled in as zero vectors (for the static plan).
    fn template_with_sizes(&self) -> Llama {
        let mut m = self.template.clone();
        for (dst, b) in m.params_mut().into_iter().zip(&self.params) {
            *dst = vec![0.0; b.len()];
        }
        m
    }

    /// Forward and backward for one batch; leaves gradients in the device buffers and returns the
    /// mean loss. `ids[i]` is followed by `targets[i]`; `batch` sequences of equal length.
    pub fn loss_and_grads(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
    ) -> Result<f32, GpuError> {
        let t = self.validate(ids, targets, batch)?;
        self.ensure_acts(batch, t)?;
        let (gid, gtg) = (
            self.gpu.upload_indices(ids, self.cfg.vocab)?,
            self.gpu.upload_indices(targets, self.cfg.vocab)?,
        );
        self.gpu.begin_batch()?;
        let rec = self.record(ids, &gid, &gtg);
        self.gpu.end_batch(rec.is_ok())?;
        rec?;
        Ok(self
            .gpu
            .read(&self.acts.as_ref().expect("allocated").loss)?[0])
    }

    /// One full optimizer step: forward, backward, clip, AdamW. The loss and the pre-clip
    /// gradient norm are the only things read back.
    pub fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, GpuError> {
        let t = self.validate(ids, targets, batch)?;
        self.ensure_acts(batch, t)?;
        let (gid, gtg) = (
            self.gpu.upload_indices(ids, self.cfg.vocab)?,
            self.gpu.upload_indices(targets, self.cfg.vocab)?,
        );
        let step = self.t + 1;
        self.gpu.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            self.record(ids, &gid, &gtg)?;
            self.optimize(step, cfg)
        })();
        self.gpu.end_batch(rec.is_ok())?;
        rec?;
        self.t = step;
        let loss = self
            .gpu
            .read(&self.acts.as_ref().expect("allocated").loss)?[0];
        let grad_norm = self.gpu.read(&self.scale)?[1];
        Ok(StepStats { loss, grad_norm })
    }

    /// Norm, clip scale and AdamW over the gradients (the recorded commands only). Absent
    /// biases are empty tensors and are skipped.
    fn optimize(&self, step: u64, cfg: &StepConfig) -> Result<(), GpuError> {
        for (i, g) in self.grads.iter().enumerate().filter(|(_, g)| !g.is_empty()) {
            sumsq_parts(self.gpu, g, &self.part, self.part_offsets[i])?;
        }
        norm_scale(self.gpu, &self.part, &self.scale, self.part.len(), cfg.clip)?;
        for (i, g) in self.grads.iter().enumerate().filter(|(_, g)| !g.is_empty()) {
            adam_update(
                self.gpu,
                (&self.params[i], g, &self.m[i], &self.v[i]),
                &self.scale,
                &cfg.adam,
                self.decays[i],
                step,
            )?;
        }
        Ok(())
    }
}
