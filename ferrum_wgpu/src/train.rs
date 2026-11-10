//! GPT-2 training on the GPU. Weights, gradients, Adam moments and every activation stay
//! resident; one step is recorded as a single command batch, and only the loss and the
//! gradient norm come back to the host.
//! ponytail: no weight tying, dropout or QAT yet, and one batch shape's activations are kept
//! at a time (reallocated when `(batch, t)` changes). Add them when a run needs them.

use crate::budget::{Budget, BufferPlan};
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::ops::{self, add};
use crate::ops_bwd as bwd;
use crate::ops_opt::{adam_update, norm_scale, parts_for, sumsq_parts};
use crate::tensor::GpuBuf;
use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::optim::Adam;

fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

/// What one optimizer step needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepConfig {
    /// AdamW hyperparameters (`lr` is this step's, already scheduled).
    pub adam: Adam,
    /// Global-norm clip cap; `0.0` disables clipping.
    pub clip: f32,
}

/// What a step reports back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    /// Mean cross-entropy of the batch (before the update).
    pub loss: f32,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
}

/// Index of block `l`'s parameter `k` (0..16) in the canonical order.
fn pi(l: usize, k: usize) -> usize {
    2 + 16 * l + k
}

/// Activations and scratch for one `(batch, t)` shape.
struct Acts {
    batch: usize,
    t: usize,
    x: Vec<GpuBuf>,
    h1: Vec<GpuBuf>,
    q: Vec<GpuBuf>,
    k: Vec<GpuBuf>,
    v: Vec<GpuBuf>,
    probs: Vec<GpuBuf>,
    a: Vec<GpuBuf>,
    x2: Vec<GpuBuf>,
    h2: Vec<GpuBuf>,
    f1: Vec<GpuBuf>,
    act: Vec<GpuBuf>,
    o: GpuBuf,
    f2: GpuBuf,
    tmp_c: GpuBuf,
    tmp_h: GpuBuf,
    hf: GpuBuf,
    logits: GpuBuf,
    dlogits: GpuBuf,
    row_loss: GpuBuf,
    loss: GpuBuf,
    dhf: GpuBuf,
    dx_a: GpuBuf,
    dx_b: GpuBuf,
    dact: GpuBuf,
    df1: GpuBuf,
    dh2: GpuBuf,
    dx2ln: GpuBuf,
    dx2: GpuBuf,
    da: GpuBuf,
    dq: GpuBuf,
    dk: GpuBuf,
    dv: GpuBuf,
    dh1a: GpuBuf,
    dh1b: GpuBuf,
    dh1c: GpuBuf,
    t1: GpuBuf,
    dh1: GpuBuf,
    dxln: GpuBuf,
    stats: GpuBuf,
    dot: GpuBuf,
    junk_v: GpuBuf,
}

/// Every activation/scratch buffer for a `(batch, t)` shape, named and sized.
pub fn activation_plan(c: &Config, batch: usize, t: usize) -> BufferPlan {
    let (r, d, h, v) = (batch * t, c.dim, c.hidden, c.vocab);
    let f = |n: usize| (n * 4) as u64;
    let mut p = BufferPlan::default();
    for l in 0..=c.layers {
        p.add(&format!("x{l}"), f(r * d));
    }
    for l in 0..c.layers {
        for (n, len) in [
            ("h1", r * d),
            ("q", r * d),
            ("k", r * d),
            ("v", r * d),
            ("probs", batch * c.heads * t * t),
            ("a", r * d),
            ("x2", r * d),
            ("h2", r * d),
            ("f1", r * h),
            ("act", r * h),
        ] {
            p.add(&format!("blk{l}.{n}"), f(len));
        }
    }
    for (n, len) in [
        ("o", r * d),
        ("f2", r * d),
        ("tmp_c", r * d),
        ("tmp_h", r * h),
        ("hf", r * d),
        ("logits", r * v),
        ("dlogits", r * v),
        ("row_loss", r),
        ("loss", 1),
        ("dhf", r * d),
        ("dx_a", r * d),
        ("dx_b", r * d),
        ("dact", r * h),
        ("df1", r * h),
        ("dh2", r * d),
        ("dx2ln", r * d),
        ("dx2", r * d),
        ("da", r * d),
        ("dq", r * d),
        ("dk", r * d),
        ("dv", r * d),
        ("dh1a", r * d),
        ("dh1b", r * d),
        ("dh1c", r * d),
        ("t1", r * d),
        ("dh1", r * d),
        ("dxln", r * d),
        ("stats", 2 * r),
        ("dot", batch * c.heads * t),
        ("junk_v", v),
    ] {
        p.add(n, f(len));
    }
    p
}

/// Buffers that live as long as the trainer: weights, gradients and both Adam moments.
pub fn static_plan(c: &Config) -> BufferPlan {
    let m = Gpt2::random(*c, 1); // shapes only
    let mut p = BufferPlan::default();
    for (i, t) in m.params().iter().enumerate() {
        for kind in ["w", "g", "m", "v"] {
            p.add(&format!("{kind}{i}"), (t.len() * 4) as u64);
        }
    }
    p
}

impl Acts {
    fn new(gpu: &Gpu, c: &Config, batch: usize, t: usize) -> Result<Acts, GpuError> {
        let (r, d, h, v) = (batch * t, c.dim, c.hidden, c.vocab);
        let z = |n: usize| gpu.zeros(n);
        let per = |n: usize| (0..c.layers).map(|_| z(n)).collect::<Result<Vec<_>, _>>();
        Ok(Acts {
            batch,
            t,
            x: (0..=c.layers).map(|_| z(r * d)).collect::<Result<_, _>>()?,
            h1: per(r * d)?,
            q: per(r * d)?,
            k: per(r * d)?,
            v: per(r * d)?,
            probs: per(batch * c.heads * t * t)?,
            a: per(r * d)?,
            x2: per(r * d)?,
            h2: per(r * d)?,
            f1: per(r * h)?,
            act: per(r * h)?,
            o: z(r * d)?,
            f2: z(r * d)?,
            tmp_c: z(r * d)?,
            tmp_h: z(r * h)?,
            hf: z(r * d)?,
            logits: z(r * v)?,
            dlogits: z(r * v)?,
            row_loss: z(r)?,
            loss: z(1)?,
            dhf: z(r * d)?,
            dx_a: z(r * d)?,
            dx_b: z(r * d)?,
            dact: z(r * h)?,
            df1: z(r * h)?,
            dh2: z(r * d)?,
            dx2ln: z(r * d)?,
            dx2: z(r * d)?,
            da: z(r * d)?,
            dq: z(r * d)?,
            dk: z(r * d)?,
            dv: z(r * d)?,
            dh1a: z(r * d)?,
            dh1b: z(r * d)?,
            dh1c: z(r * d)?,
            t1: z(r * d)?,
            dh1: z(r * d)?,
            dxln: z(r * d)?,
            stats: z(2 * r)?,
            dot: z(batch * c.heads * t)?,
            junk_v: z(v)?,
        })
    }
}

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

    /// One full optimizer step: forward, backward, clip, AdamW. One command batch; the loss and
    /// the pre-clip gradient norm are the only things read back.
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
            for (i, g) in self.grads.iter().enumerate() {
                sumsq_parts(self.gpu, g, &self.part, self.part_offsets[i])?;
            }
            norm_scale(self.gpu, &self.part, &self.scale, self.part.len(), cfg.clip)?;
            for i in 0..self.params.len() {
                adam_update(
                    self.gpu,
                    (&self.params[i], &self.grads[i], &self.m[i], &self.v[i]),
                    &self.scale,
                    &cfg.adam,
                    self.decays[i],
                    step,
                )?;
            }
            Ok(())
        })();
        self.gpu.end_batch(rec.is_ok())?;
        rec?;
        self.t = step;
        let loss = self
            .gpu
            .read(&self.acts.as_ref().expect("allocated").loss)?[0];
        let norm = self.gpu.read(&self.scale)?[1];
        Ok(StepStats {
            loss,
            grad_norm: norm,
        })
    }

    /// Record the forward and backward dispatches (a batch must be open).
    fn record(&self, ids_host: &[usize], ids: &GpuBuf, targets: &GpuBuf) -> Result<(), GpuError> {
        let a = self.acts.as_ref().expect("allocated");
        let (g, c, p, gr) = (self.gpu, &self.cfg, &self.params, &self.grads);
        let (b, t, nl) = (a.batch, a.t, c.layers);
        let (r, d, h, v, hd) = (b * t, c.dim, c.hidden, c.vocab, c.head_dim());
        let lin = |x: &GpuBuf,
                   w: &GpuBuf,
                   bias: &GpuBuf,
                   y: &GpuBuf,
                   tmp: &GpuBuf,
                   (k, n): (usize, usize)|
         -> Result<(), GpuError> {
            self.matmul.submit(g, x, w, tmp, (r, k, n))?;
            ops::add_bias(g, tmp, bias, y)
        };
        // Forward.
        ops::embed(g, (&p[0], &p[1], ids), &a.x[0], (r, d, t))?;
        for l in 0..nl {
            ops::layernorm(g, &a.x[l], &p[pi(l, 0)], &p[pi(l, 1)], &a.h1[l], (r, d))?;
            lin(
                &a.h1[l],
                &p[pi(l, 2)],
                &p[pi(l, 3)],
                &a.q[l],
                &a.tmp_c,
                (d, d),
            )?;
            lin(
                &a.h1[l],
                &p[pi(l, 4)],
                &p[pi(l, 5)],
                &a.k[l],
                &a.tmp_c,
                (d, d),
            )?;
            lin(
                &a.h1[l],
                &p[pi(l, 6)],
                &p[pi(l, 7)],
                &a.v[l],
                &a.tmp_c,
                (d, d),
            )?;
            ops::attention(
                g,
                (&a.q[l], &a.k[l], &a.v[l]),
                (&a.probs[l], &a.a[l]),
                (b, t, c.heads, hd),
            )?;
            lin(&a.a[l], &p[pi(l, 8)], &p[pi(l, 9)], &a.o, &a.tmp_c, (d, d))?;
            add(g, &a.x[l], &a.o, &a.x2[l])?;
            ops::layernorm(g, &a.x2[l], &p[pi(l, 10)], &p[pi(l, 11)], &a.h2[l], (r, d))?;
            lin(
                &a.h2[l],
                &p[pi(l, 12)],
                &p[pi(l, 13)],
                &a.f1[l],
                &a.tmp_h,
                (d, h),
            )?;
            ops::gelu(g, &a.f1[l], &a.act[l])?;
            lin(
                &a.act[l],
                &p[pi(l, 14)],
                &p[pi(l, 15)],
                &a.f2,
                &a.tmp_c,
                (h, d),
            )?;
            add(g, &a.x2[l], &a.f2, &a.x[l + 1])?;
        }
        let fb = pi(nl, 0);
        ops::layernorm(g, &a.x[nl], &p[fb], &p[fb + 1], &a.hf, (r, d))?;
        self.matmul
            .submit(g, &a.hf, &p[fb + 2], &a.logits, (r, d, v))?;
        ops::cross_entropy(
            g,
            &a.logits,
            targets,
            (&a.dlogits, &a.row_loss, &a.loss),
            (r, v),
        )?;
        // Backward.
        bwd::linear_bwd(
            g,
            (&a.hf, &p[fb + 2], &a.dlogits),
            (&a.dhf, &gr[fb + 2], &a.junk_v),
            (r, d, v),
        )?;
        bwd::layernorm_bwd(
            g,
            (&a.x[nl], &p[fb], &a.dhf),
            (&a.dx_a, &gr[fb], &gr[fb + 1]),
            &a.stats,
            (r, d),
        )?;
        let (mut cur, mut next) = (&a.dx_a, &a.dx_b);
        for l in (0..nl).rev() {
            bwd::linear_bwd(
                g,
                (&a.act[l], &p[pi(l, 14)], cur),
                (&a.dact, &gr[pi(l, 14)], &gr[pi(l, 15)]),
                (r, h, d),
            )?;
            bwd::gelu_bwd(g, &a.dact, &a.f1[l], &a.df1)?;
            bwd::linear_bwd(
                g,
                (&a.h2[l], &p[pi(l, 12)], &a.df1),
                (&a.dh2, &gr[pi(l, 12)], &gr[pi(l, 13)]),
                (r, d, h),
            )?;
            bwd::layernorm_bwd(
                g,
                (&a.x2[l], &p[pi(l, 10)], &a.dh2),
                (&a.dx2ln, &gr[pi(l, 10)], &gr[pi(l, 11)]),
                &a.stats,
                (r, d),
            )?;
            add(g, cur, &a.dx2ln, &a.dx2)?;
            bwd::linear_bwd(
                g,
                (&a.a[l], &p[pi(l, 8)], &a.dx2),
                (&a.da, &gr[pi(l, 8)], &gr[pi(l, 9)]),
                (r, d, d),
            )?;
            bwd::attention_bwd(
                g,
                (&a.q[l], &a.k[l], &a.v[l], &a.probs[l], &a.da),
                (&a.dq, &a.dk, &a.dv),
                &a.dot,
                (b, t, c.heads, hd),
            )?;
            bwd::linear_bwd(
                g,
                (&a.h1[l], &p[pi(l, 2)], &a.dq),
                (&a.dh1a, &gr[pi(l, 2)], &gr[pi(l, 3)]),
                (r, d, d),
            )?;
            bwd::linear_bwd(
                g,
                (&a.h1[l], &p[pi(l, 4)], &a.dk),
                (&a.dh1b, &gr[pi(l, 4)], &gr[pi(l, 5)]),
                (r, d, d),
            )?;
            bwd::linear_bwd(
                g,
                (&a.h1[l], &p[pi(l, 6)], &a.dv),
                (&a.dh1c, &gr[pi(l, 6)], &gr[pi(l, 7)]),
                (r, d, d),
            )?;
            add(g, &a.dh1a, &a.dh1b, &a.t1)?;
            add(g, &a.t1, &a.dh1c, &a.dh1)?;
            bwd::layernorm_bwd(
                g,
                (&a.x[l], &p[pi(l, 0)], &a.dh1),
                (&a.dxln, &gr[pi(l, 0)], &gr[pi(l, 1)]),
                &a.stats,
                (r, d),
            )?;
            add(g, &a.dx2, &a.dxln, next)?;
            std::mem::swap(&mut cur, &mut next);
        }
        bwd::embed_bwd(g, cur, ids_host, (&gr[0], &gr[1]), (v, d, t, c.ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Source;
    use crate::device::open_for_test;
    use crate::tolerance::max_rel_err;
    use ferrum_ref::grad::loss_and_grads;
    use ferrum_ref::optim::{adam_step, clip_scale, global_norm, AdamState};

    fn cfg() -> Config {
        Config {
            vocab: 13,
            ctx: 8,
            dim: 8,
            heads: 2,
            layers: 2,
            hidden: 16,
        }
    }

    fn batch_data(c: &Config, batch: usize, t: usize, salt: usize) -> (Vec<usize>, Vec<usize>) {
        let n = batch * t;
        (
            (0..n).map(|i| (i * 3 + 1 + salt) % c.vocab).collect(),
            (0..n).map(|i| (i * 5 + 2 + salt) % c.vocab).collect(),
        )
    }

    #[test]
    fn plans_are_consistent_with_the_allocated_buffers() {
        let c = cfg();
        let sp = static_plan(&c);
        assert_eq!(sp.rows().len(), 4 * (2 + 2 * 16 + 3));
        let w = Gpt2::random(c, 1);
        assert_eq!(sp.total(), 4 * 4 * w.num_params() as u64);
        let ap = activation_plan(&c, 2, 5);
        assert!(ap.rows().iter().any(|(n, _)| n == "blk1.probs"));
        assert!(activation_plan(&c, 4, 5).total() > ap.total());
    }

    #[test]
    fn weights_round_trip_through_the_device() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 2);
        let tr = GpuTrainer::new(&g, &w, None).unwrap();
        assert_eq!(tr.weights_host().unwrap(), w);
        assert_eq!(tr.config(), &w.cfg);
        assert_eq!(tr.steps(), 0);
        let (m, v) = tr.moments_host().unwrap();
        assert_eq!(m, w.zeros_like());
        assert_eq!(v, w.zeros_like());
    }

    #[test]
    fn loss_and_every_gradient_match_the_cpu_backward() {
        let Some(g) = open_for_test() else { return };
        let c = cfg();
        let w = Gpt2::random(c, 3);
        let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
        for (batch, t) in [(1usize, 1usize), (1, 3), (2, 5), (3, 8)] {
            let (ids, targets) = batch_data(&c, batch, t, batch + t);
            let (want_loss, want) = loss_and_grads(&w, &ids, &targets, batch, 1).unwrap();
            let loss = tr.loss_and_grads(&ids, &targets, batch).unwrap();
            assert!(
                (loss - want_loss).abs() < 1e-5 * want_loss.max(1.0),
                "{batch}x{t}: loss {loss} vs {want_loss}"
            );
            let got = tr.grads_host().unwrap();
            let mut worst = 0.0f32;
            for (i, (gp, wp)) in got.params().iter().zip(want.params()).enumerate() {
                let e = max_rel_err(gp, wp);
                worst = worst.max(e);
                assert!(
                    e < crate::tolerance::BACKWARD,
                    "{batch}x{t} param {i}: max rel err {e}"
                );
            }
            eprintln!("grads {batch}x{t}: worst max rel err {worst:.2e}");
        }
    }

    #[test]
    fn optimizer_steps_track_the_cpu_trainer() {
        let Some(g) = open_for_test() else { return };
        let c = cfg();
        let w0 = Gpt2::init(c, 4);
        let adam = Adam {
            weight_decay: 0.01,
            ..Adam::new(0.01)
        };
        let mut cpu = w0.clone();
        let mut st = AdamState::new(&w0);
        let mut tr = GpuTrainer::new(&g, &w0, None).unwrap();
        for s in 0..5usize {
            let (ids, targets) = batch_data(&c, 2, 6, s);
            let (cpu_loss, grads) = loss_and_grads(&cpu, &ids, &targets, 2, 1).unwrap();
            let norm = global_norm(&grads);
            adam_step(&mut cpu, &grads, &mut st, &adam, clip_scale(norm, 0.05));
            let stats = tr
                .step(&ids, &targets, 2, &StepConfig { adam, clip: 0.05 })
                .unwrap();
            assert!(
                (stats.loss - cpu_loss).abs() < 1e-4 * cpu_loss.max(1.0),
                "step {s}: loss {} vs {cpu_loss}",
                stats.loss
            );
            assert!(
                (stats.grad_norm - norm).abs() < 1e-3 * norm.max(1.0),
                "step {s}: norm {} vs {norm}",
                stats.grad_norm
            );
        }
        assert_eq!(tr.steps(), 5);
        let got = tr.weights_host().unwrap();
        let mut worst = 0.0f32;
        for (i, (gp, wp)) in got.params().iter().zip(cpu.params()).enumerate() {
            let e = max_rel_err(gp, wp);
            worst = worst.max(e);
            assert!(
                e < crate::tolerance::BACKWARD,
                "weights after 5 steps, param {i}: max rel err {e}"
            );
        }
        eprintln!("weights after 5 clipped AdamW steps: worst max rel err {worst:.2e}");
    }

    #[test]
    fn training_a_tiny_model_on_the_gpu_reduces_the_loss() {
        let Some(g) = open_for_test() else { return };
        let c = cfg();
        let (ids, targets) = batch_data(&c, 2, 6, 0);
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 5), None).unwrap();
        let sc = StepConfig {
            adam: Adam::new(0.01),
            clip: 1.0,
        };
        let first = tr.step(&ids, &targets, 2, &sc).unwrap().loss;
        let mut last = first;
        for _ in 0..60 {
            last = tr.step(&ids, &targets, 2, &sc).unwrap().loss;
        }
        assert!(first > 2.0, "starts near ln(13) = 2.56, got {first}");
        assert!(last < first * 0.4, "memorises one batch: {first} -> {last}");
    }

    #[test]
    fn two_identical_runs_end_with_identical_bytes() {
        let Some(g) = open_for_test() else { return };
        let c = cfg();
        let run = || {
            let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 6), None).unwrap();
            for s in 0..4usize {
                let (ids, targets) = batch_data(&c, 2, 5, s);
                tr.step(
                    &ids,
                    &targets,
                    2,
                    &StepConfig {
                        adam: Adam::new(0.01),
                        clip: 0.5,
                    },
                )
                .unwrap();
            }
            (tr.weights_host().unwrap(), tr.moments_host().unwrap())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn load_restores_weights_moments_and_the_step_counter_exactly() {
        let Some(g) = open_for_test() else { return };
        let c = cfg();
        let (ids, targets) = batch_data(&c, 2, 4, 1);
        let sc = StepConfig {
            adam: Adam::new(0.01),
            clip: 0.0,
        };
        // Train 3 steps straight through.
        let mut a = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
        for _ in 0..3 {
            a.step(&ids, &targets, 2, &sc).unwrap();
        }
        // Train 2, checkpoint to the host, resume in a fresh trainer for the third.
        let mut b = GpuTrainer::new(&g, &Gpt2::init(c, 7), None).unwrap();
        for _ in 0..2 {
            b.step(&ids, &targets, 2, &sc).unwrap();
        }
        let (w, (m, v), t) = (
            b.weights_host().unwrap(),
            b.moments_host().unwrap(),
            b.steps(),
        );
        let mut resumed = GpuTrainer::new(&g, &Gpt2::init(c, 99), None).unwrap();
        resumed.load(&w, Some((&m, &v)), t).unwrap();
        resumed.step(&ids, &targets, 2, &sc).unwrap();
        assert_eq!(resumed.steps(), 3);
        assert_eq!(
            resumed.weights_host().unwrap(),
            a.weights_host().unwrap(),
            "resume = uninterrupted"
        );
        assert_eq!(resumed.moments_host().unwrap(), a.moments_host().unwrap());
        // A different shape is refused.
        let other = Gpt2::init(Config { dim: 12, ..c }, 1);
        assert!(resumed
            .load(&other, None, 0)
            .unwrap_err()
            .to_string()
            .contains("shape"));
    }

    #[test]
    fn batch_shapes_can_change_between_steps() {
        let Some(g) = open_for_test() else { return };
        let c = cfg();
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 8), None).unwrap();
        let sc = StepConfig {
            adam: Adam::new(0.005),
            clip: 1.0,
        };
        for (b, t) in [(2usize, 5usize), (1, 3), (2, 5), (4, 2)] {
            let (ids, targets) = batch_data(&c, b, t, 0);
            let s = tr.step(&ids, &targets, b, &sc).unwrap();
            assert!(s.loss.is_finite() && s.grad_norm.is_finite(), "{b}x{t}");
        }
    }

    #[test]
    fn bad_input_and_oversized_batches_are_refused_before_allocating() {
        let Some(g) = open_for_test() else { return };
        let c = cfg();
        let w = Gpt2::init(c, 9);
        let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
        let sc = StepConfig {
            adam: Adam::new(0.01),
            clip: 0.0,
        };
        let mut e = |ids: &[usize], tg: &[usize], b: usize| {
            tr.step(ids, tg, b, &sc).unwrap_err().to_string()
        };
        assert!(e(&[], &[], 1).contains("split"));
        assert!(e(&[1, 2, 3], &[1, 2, 3], 2).contains("split"));
        assert!(e(&[1; 9], &[1; 9], 1).contains("exceeds the context"));
        assert!(e(&[1, 2], &[1], 1).contains("targets"));
        assert!(e(&[13], &[1], 1).contains("out of range"));
        assert_eq!(tr.steps(), 0, "failed steps do not count");
        // A tight budget: the static part fits, a big batch does not.
        let need = static_plan(&c).total();
        let tight = Some(Budget {
            bytes: need + 1000,
            source: Source::Override,
        });
        let mut small = GpuTrainer::new(&g, &w, tight).unwrap();
        let before = g.buffers_created();
        let (ids, targets) = batch_data(&c, 4, 8, 0);
        let msg = small.step(&ids, &targets, 4, &sc).unwrap_err().to_string();
        assert!(
            msg.contains("does not fit") && msg.contains("budget"),
            "{msg}"
        );
        assert_eq!(
            g.buffers_created(),
            before,
            "refused before any activation buffer exists"
        );
        // And the static part alone can be refused too.
        let tiny = Some(Budget {
            bytes: 100,
            source: Source::Override,
        });
        assert!(GpuTrainer::new(&g, &w, tiny).is_err());
    }
}
