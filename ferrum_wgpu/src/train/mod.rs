//! GPT-2 training on the GPU. Weights, gradients, Adam moments and every activation stay
//! resident; one step is recorded as a single command batch, and only the loss and the
//! gradient norm come back to the host.
//! ponytail: no weight tying, dropout or QAT yet, and one batch shape's activations are kept
//! at a time (reallocated when `(batch, t)` changes). Add them when a run needs them.

mod acts;
mod engine;
mod plan;
mod record;
#[cfg(test)]
mod tests;

use crate::budget::{Budget, BufferPlan};
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::ops_opt::{accumulate, adam_update, norm_scale, parts_for, sumsq_parts};
use crate::tensor::GpuBuf;
use acts::Acts;
use ferrum_ref::gpt2::{Config, Gpt2};

pub use plan::{accum_plan, activation_plan, static_plan};
fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

pub use ferrum_ref::trainer::{Engine, Snapshot, StepConfig, StepStats};

/// A GPT-2 model being trained on the GPU.
pub struct GpuTrainer<'g> {
    gpu: &'g Gpu,
    cfg: Config,
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
    accum: Option<Vec<GpuBuf>>,
    last_micro: usize,
    t: u64,
}

impl<'g> GpuTrainer<'g> {
    /// Upload `w`. Weights, gradients and moments are planned and checked against the adapter
    /// and `budget` before anything is allocated; activations are checked when a batch shape is
    /// first seen (including the static part, so the whole footprint is bounded).
    pub fn new(gpu: &'g Gpu, w: &Gpt2, budget: Option<Budget>) -> Result<Self, GpuError> {
        w.cfg.validate().map_err(err)?;
        static_plan(&w.cfg)
            .check(gpu.info().max_binding, budget)
            .map_err(|e| err(e.to_string()))?;
        let up = |m: &Gpt2| {
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
        Ok(GpuTrainer {
            gpu,
            cfg: w.cfg,
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
            accum: None,
            last_micro: 0,
            t: 0,
        })
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Optimizer steps taken.
    pub fn steps(&self) -> u64 {
        self.t
    }

    fn host(&self, bufs: &[GpuBuf]) -> Result<Gpt2, GpuError> {
        let mut out = Gpt2::random(self.cfg, 1);
        for (dst, b) in out.params_mut().into_iter().zip(bufs) {
            *dst = self.gpu.read(b)?;
        }
        Ok(out)
    }

    /// Copy the weights back to the host.
    pub fn weights_host(&self) -> Result<Gpt2, GpuError> {
        self.host(&self.params)
    }

    /// Copy the gradients of the last forward/backward back to the host.
    pub fn grads_host(&self) -> Result<Gpt2, GpuError> {
        self.host(&self.grads)
    }

    /// Copy the Adam moments back to the host as `(m, v)`.
    pub fn moments_host(&self) -> Result<(Gpt2, Gpt2), GpuError> {
        Ok((self.host(&self.m)?, self.host(&self.v)?))
    }

    /// Replace weights, Adam moments and the step counter (resume from a checkpoint).
    pub fn load(
        &mut self,
        w: &Gpt2,
        moments: Option<(&Gpt2, &Gpt2)>,
        steps: u64,
    ) -> Result<(), GpuError> {
        if w.cfg != self.cfg {
            return Err(err("load: the model shape differs from the trainer's"));
        }
        let (zm, zv) = (w.zeros_like(), w.zeros_like());
        let (m, v) = moments.unwrap_or((&zm, &zv));
        if m.cfg != self.cfg || v.cfg != self.cfg {
            return Err(err("load: the moment shapes differ from the trainer's"));
        }
        for (set, src) in [(&self.params, w), (&self.m, m), (&self.v, v)] {
            for (b, p) in set.iter().zip(src.params()) {
                self.gpu.copy_from_host(b, p)?;
            }
        }
        self.t = steps;
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
        let mut plan = static_plan(&self.cfg);
        activation_plan(&self.cfg, batch, t)
            .rows()
            .iter()
            .for_each(|(n, b)| plan.add(n, *b));
        plan.check(self.gpu.info().max_binding, self.budget)
            .map_err(|e| err(format!("batch {batch} x {t} tokens does not fit: {e}")))?;
        self.acts = Some(Acts::new(self.gpu, &self.cfg, batch, t)?);
        Ok(())
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
    ///
    /// If the whole batch does not fit in the memory budget it is split into equal micro-batches
    /// (the largest divisor of `batch` that fits), whose gradients are accumulated on the device
    /// before a single optimizer step. The result matches the unsplit step up to floating-point
    /// summation order. Only when even one sequence at a time does not fit is the step refused.
    pub fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, GpuError> {
        let t = self.validate(ids, targets, batch)?;
        let micro = self.choose_micro(batch, t)?;
        self.step_split(ids, targets, (batch, micro, t), cfg)
    }

    /// Like [`step`](Self::step) but with an explicit micro-batch size, which must divide `batch`.
    pub fn step_with_micro(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        micro: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, GpuError> {
        let t = self.validate(ids, targets, batch)?;
        if micro == 0 || !batch.is_multiple_of(micro) {
            return Err(err(format!(
                "micro-batch {micro} must be positive and divide the batch of {batch}"
            )));
        }
        self.check_fits(batch, micro, t)?;
        self.step_split(ids, targets, (batch, micro, t), cfg)
    }

    /// Weights and Adam moments together, for tests that compare whole training states.
    #[cfg(test)]
    pub(crate) fn snapshot_pair(&self) -> (Gpt2, (Gpt2, Gpt2)) {
        (self.weights_host().unwrap(), self.moments_host().unwrap())
    }

    /// Sequences per micro-batch used by the last step (equals the batch when it was not split).
    pub fn last_micro(&self) -> usize {
        self.last_micro
    }

    /// The memory plan of running `batch` sequences as micro-batches of `micro`.
    fn footprint(&self, batch: usize, micro: usize, t: usize) -> BufferPlan {
        let mut plan = static_plan(&self.cfg);
        if micro < batch {
            accum_plan(&self.cfg)
                .rows()
                .iter()
                .for_each(|(n, b)| plan.add(n, *b));
        }
        activation_plan(&self.cfg, micro, t)
            .rows()
            .iter()
            .for_each(|(n, b)| plan.add(n, *b));
        plan
    }

    fn check_fits(&self, batch: usize, micro: usize, t: usize) -> Result<(), GpuError> {
        self.footprint(batch, micro, t)
            .check(self.gpu.info().max_binding, self.budget)
            .map_err(|e| {
                err(format!(
                    "batch {batch} x {t} tokens does not fit in micro-batches of {micro}: {e}"
                ))
            })
    }

    /// The largest micro-batch size (a divisor of `batch`) whose footprint fits. When nothing
    /// fits, the error is the one for a single sequence, the closest to fitting.
    fn choose_micro(&self, batch: usize, t: usize) -> Result<usize, GpuError> {
        let mut last = err("empty batch");
        for m in (1..=batch).rev().filter(|m| batch.is_multiple_of(*m)) {
            match self.check_fits(batch, m, t) {
                Ok(()) => return Ok(m),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn step_split(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        (batch, micro, t): (usize, usize, usize),
        cfg: &StepConfig,
    ) -> Result<StepStats, GpuError> {
        let chunks = batch / micro;
        self.ensure_acts(micro, t)?;
        if chunks > 1 {
            self.ensure_accum()?;
        }
        let rows = micro * t;
        let (mut gid, mut gtg) = (Vec::with_capacity(chunks), Vec::with_capacity(chunks));
        for c in 0..chunks {
            let r = c * rows..(c + 1) * rows;
            gid.push(self.gpu.upload_indices(&ids[r.clone()], self.cfg.vocab)?);
            gtg.push(self.gpu.upload_indices(&targets[r], self.cfg.vocab)?);
        }
        let losses = self.gpu.zeros(chunks)?;
        let step = self.t + 1;
        self.gpu.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            let acts = self.acts.as_ref().expect("allocated");
            let weight = 1.0 / chunks as f32;
            for c in 0..chunks {
                self.record(&ids[c * rows..(c + 1) * rows], &gid[c], &gtg[c])?;
                self.gpu.copy_f32(&acts.loss, 0, &losses, c, 1)?;
                if let Some(acc) = self.accum.as_ref().filter(|_| chunks > 1) {
                    for (a, g) in acc.iter().zip(&self.grads) {
                        accumulate(self.gpu, a, g, weight, c == 0)?;
                    }
                }
            }
            let source = match self.accum.as_ref() {
                Some(acc) if chunks > 1 => acc,
                _ => &self.grads,
            };
            self.optimize(source, step, cfg)
        })();
        self.gpu.end_batch(rec.is_ok())?;
        rec?;
        self.t = step;
        self.last_micro = micro;
        let per_chunk = self.gpu.read(&losses)?;
        let loss = per_chunk.iter().sum::<f32>() / chunks as f32;
        let norm = self.gpu.read(&self.scale)?[1];
        Ok(StepStats {
            loss,
            grad_norm: norm,
        })
    }

    /// Norm, clip scale and AdamW over the gradients in `source` (the recorded commands only).
    fn optimize(&self, source: &[GpuBuf], step: u64, cfg: &StepConfig) -> Result<(), GpuError> {
        for (i, g) in source.iter().enumerate() {
            sumsq_parts(self.gpu, g, &self.part, self.part_offsets[i])?;
        }
        norm_scale(self.gpu, &self.part, &self.scale, self.part.len(), cfg.clip)?;
        for (i, g) in source.iter().enumerate() {
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

    /// Allocate the gradient accumulators (once, on first use of micro-batching).
    fn ensure_accum(&mut self) -> Result<(), GpuError> {
        if self.accum.is_none() {
            let bufs = self
                .params
                .iter()
                .map(|p| self.gpu.zeros(p.len()))
                .collect::<Result<Vec<_>, _>>()?;
            self.accum = Some(bufs);
        }
        Ok(())
    }
}
