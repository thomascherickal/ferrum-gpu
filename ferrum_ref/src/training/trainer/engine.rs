// Provenance: the window/epoch/step design (windows at a stride, a fresh shuffle each epoch
// drawn without replacement, step budget, warmup+cosine, checkpoint cadence) follows ferrum-cpu
// `slm/training_loop.rs` and `train_transformer/epoch.rs` (commit 56a9d3a). Our own code.
//! The step interface ([`Engine`]), what a step needs and reports, and the CPU engine.

use crate::dropout::{step_seed, Dropout};
use crate::gpt2::Gpt2;
use crate::grad::loss_and_grads_with;
use crate::optim::{adam_step, clip_scale, global_norm, Adam, AdamState};

/// What one optimizer step needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepConfig {
    /// AdamW hyperparameters (`lr` is this step's, already scheduled).
    pub adam: Adam,
    /// Global-norm clip cap; `0.0` disables clipping.
    pub clip: f32,
}

/// What a step reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    /// Mean cross-entropy of the batch (before the update).
    pub loss: f32,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
}

/// Everything an engine needs to continue a run: weights, Adam moments, steps taken.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// Weights.
    pub weights: Gpt2,
    /// First moments.
    pub m: Gpt2,
    /// Second moments.
    pub v: Gpt2,
    /// Optimizer steps completed.
    pub steps: u64,
}

/// A backend that can run training steps.
pub trait Engine {
    /// One forward/backward/clip/AdamW step on `batch` equal-length sequences.
    fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, String>;
    /// Optimizer steps completed.
    fn steps(&self) -> u64;
    /// Copy the full training state to the host.
    fn snapshot(&self) -> Result<Snapshot, String>;
    /// Replace the full training state.
    fn restore(&mut self, s: &Snapshot) -> Result<(), String>;
}

/// The CPU engine (also the oracle for the GPU engine).
pub struct CpuTrainer {
    w: Gpt2,
    state: AdamState,
    threads: usize,
    qat: bool,
    tie: bool,
    /// FFN-hidden dropout rate and the run's base seed (rate 0 is off).
    dropout: (f32, u64),
}

impl CpuTrainer {
    /// A trainer for `w` with `threads` matmul threads (the result does not depend on it).
    pub fn new(w: Gpt2, threads: usize) -> Self {
        let state = AdamState::new(&w);
        CpuTrainer {
            w,
            state,
            threads,
            qat: false,
            tie: false,
            dropout: (0.0, 0),
        }
    }

    /// Train with FFN-hidden dropout at `rate` (clamped to 0.95; 0 or less is off). Each step's
    /// mask is a function of `seed` and the step number, so a resumed run continues the same
    /// masks and two runs with the same seed agree exactly.
    pub fn with_dropout(mut self, rate: f32, seed: u64) -> Self {
        self.dropout = (rate, seed);
        self
    }

    /// Train with the output head tied to the token embedding (`head = tokᵀ`, one shared matrix).
    /// The head is tied at once, and stays tied after every step.
    pub fn with_tie(mut self, on: bool) -> Self {
        self.tie = on;
        if on {
            self.w.tie_head();
        }
        self
    }

    /// Turn quantisation-aware training on or off (off by default).
    pub fn with_qat(mut self, on: bool) -> Self {
        self.qat = on;
        self
    }

    /// The current (master) weights.
    pub fn weights(&self) -> &Gpt2 {
        &self.w
    }
}

impl Engine for CpuTrainer {
    fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        cfg: &StepConfig,
    ) -> Result<StepStats, String> {
        // QAT: gradients are taken at the Q8_0-snapped weights, but Adam updates the masters.
        let snapped;
        let model = if self.qat {
            let mut s = self.w.fake_quantize();
            if self.tie {
                s.tie_head(); // the head is the snapped embedding, as the exported file will have it
            }
            snapped = s;
            &snapped
        } else {
            &self.w
        };
        let drop = Dropout::new(self.dropout.0, step_seed(self.dropout.1, self.state.t));
        let (loss, mut g) = loss_and_grads_with(model, ids, targets, batch, self.threads, drop)?;
        if self.tie {
            g.fold_head_grad();
        }
        let norm = global_norm(&g);
        adam_step(
            &mut self.w,
            &g,
            &mut self.state,
            &cfg.adam,
            clip_scale(norm, cfg.clip),
        );
        if self.tie {
            self.w.tie_head(); // Adam only moved the embedding (the head's gradient was zero)
        }
        Ok(StepStats {
            loss,
            grad_norm: norm,
        })
    }

    fn steps(&self) -> u64 {
        self.state.t
    }

    fn snapshot(&self) -> Result<Snapshot, String> {
        Ok(Snapshot {
            weights: self.w.clone(),
            m: self.state.m.clone(),
            v: self.state.v.clone(),
            steps: self.state.t,
        })
    }

    fn restore(&mut self, s: &Snapshot) -> Result<(), String> {
        if s.weights.cfg != self.w.cfg || s.m.cfg != self.w.cfg || s.v.cfg != self.w.cfg {
            return Err("restore: the model shape differs from the trainer's".into());
        }
        self.w = s.weights.clone();
        if self.tie {
            self.w.tie_head(); // a snapshot from an untied run is tied on the way in
        }
        self.state = AdamState {
            m: s.m.clone(),
            v: s.v.clone(),
            t: s.steps,
        };
        Ok(())
    }
}
