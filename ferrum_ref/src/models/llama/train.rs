// Provenance: AdamW update order, decoupled decay and global-norm clipping are the ones in
// `crate::optim` (copied there from ferrum-cpu commit 56a9d3a); duplicated over [`Llama`] rather
// than generalised because `Gpt2` is concrete in the optimizer and the loop.
// ponytail: duplicated; make `optim` generic over a `params()` trait when a third model appears.
//! AdamW and a single-step trainer for [`Llama`] on the CPU: the oracle for the GPU trainer.

use super::grad::loss_and_grads;
use super::Llama;
use crate::optim::{clip_scale, Adam};
use crate::trainer::{StepConfig, StepStats};

/// First and second moments plus the step counter.
#[derive(Clone, Debug, PartialEq)]
pub struct Moments {
    /// First moments, same shape as the model.
    pub m: Llama,
    /// Second moments.
    pub v: Llama,
    /// Completed steps.
    pub t: u64,
}

impl Moments {
    /// Zero moments for `w`.
    pub fn new(w: &Llama) -> Self {
        Moments {
            m: w.zeros_like(),
            v: w.zeros_like(),
            t: 0,
        }
    }
}

/// Global L2 norm of all gradients (accumulated in f64).
pub fn global_norm(g: &Llama) -> f32 {
    let s: f64 = g
        .params()
        .iter()
        .flat_map(|p| p.iter())
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    s.sqrt() as f32
}

/// One AdamW step (increments `state.t`); `grad_scale` multiplies every gradient first.
pub fn adam_step(w: &mut Llama, g: &Llama, state: &mut Moments, adam: &Adam, grad_scale: f32) {
    state.t += 1;
    let (bc1, bc2) = adam.bias_corrections(state.t);
    let decays: Vec<bool> = (0..w.params().len()).map(|i| w.decays(i)).collect();
    let (gs, ms, vs) = (g.params(), state.m.params_mut(), state.v.params_mut());
    for (i, ((p, (gp, mp)), vp)) in w
        .params_mut()
        .into_iter()
        .zip(gs.into_iter().zip(ms))
        .zip(vs)
        .enumerate()
    {
        let wd = if decays[i] { adam.weight_decay } else { 0.0 };
        for j in 0..p.len() {
            let gr = gp[j] * grad_scale;
            mp[j] = adam.beta1 * mp[j] + (1.0 - adam.beta1) * gr;
            vp[j] = adam.beta2 * vp[j] + (1.0 - adam.beta2) * gr * gr;
            let (mh, vh) = (mp[j] / bc1, vp[j] / bc2);
            let mut update = adam.lr * mh / (vh.sqrt() + adam.eps);
            if wd != 0.0 {
                update += adam.lr * wd * p[j];
            }
            p[j] -= update;
        }
    }
}

/// The CPU Llama trainer (also the oracle for the GPU trainer).
pub struct LlamaTrainer {
    w: Llama,
    state: Moments,
    threads: usize,
}

impl LlamaTrainer {
    /// A trainer for `w` with `threads` matmul threads (the result does not depend on it).
    pub fn new(w: Llama, threads: usize) -> Self {
        let state = Moments::new(&w);
        LlamaTrainer { w, state, threads }
    }

    /// The current (master) weights.
    pub fn weights(&self) -> &Llama {
        &self.w
    }

    /// Optimizer steps completed.
    pub fn steps(&self) -> u64 {
        self.state.t
    }

    /// One forward/backward/clip/AdamW step on `batch` equal-length sequences.
    pub fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, String> {
        let (loss, g) = loss_and_grads(&self.w, ids, targets, batch, self.threads)?;
        let norm = global_norm(&g);
        adam_step(
            &mut self.w,
            &g,
            &mut self.state,
            &cfg.adam,
            clip_scale(norm, cfg.clip),
        );
        Ok(StepStats {
            loss,
            grad_norm: norm,
        })
    }
}
