// Provenance: Adam/AdamW update order and decoupled-decay rule, global-norm clipping and the
// warmup+cosine schedule follow ferrum-cpu `optim.rs` and `train_transformer/param.rs`
// (commit 56a9d3a); rewritten over plain slices, ours from here on.
//! AdamW, gradient clipping and the learning-rate schedule on the CPU (the oracle for the GPU
//! optimizer kernels).

use crate::gpt2::Gpt2;

/// Adam/AdamW hyperparameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adam {
    /// Learning rate for this step.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Denominator epsilon.
    pub eps: f32,
    /// Decoupled weight decay (applied to weight matrices and embeddings only).
    pub weight_decay: f32,
}

impl Adam {
    /// Standard defaults: beta1 0.9, beta2 0.999, eps 1e-8, no weight decay.
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Bias corrections `(1 - beta1^t, 1 - beta2^t)`; computed on the host for every backend.
    pub fn bias_corrections(&self, t: u64) -> (f32, f32) {
        let t = t.min(i32::MAX as u64) as i32;
        (1.0 - self.beta1.powi(t), 1.0 - self.beta2.powi(t))
    }
}

/// First and second moments plus the step counter.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamState {
    /// First moments, same shape as the model.
    pub m: Gpt2,
    /// Second moments.
    pub v: Gpt2,
    /// Completed steps.
    pub t: u64,
}

impl AdamState {
    /// Zero moments for `w`.
    pub fn new(w: &Gpt2) -> Self {
        AdamState {
            m: w.zeros_like(),
            v: w.zeros_like(),
            t: 0,
        }
    }
}

/// Global L2 norm of all gradients (accumulated in f64 like ferrum-cpu).
pub fn global_norm(g: &Gpt2) -> f32 {
    let s: f64 = g
        .params()
        .iter()
        .flat_map(|p| p.iter())
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    s.sqrt() as f32
}

/// The factor gradients are multiplied by to cap their global norm at `max_norm`
/// (1.0 when `max_norm <= 0`, the norm is within the cap, or the norm is not finite).
pub fn clip_scale(norm: f32, max_norm: f32) -> f32 {
    if max_norm > 0.0 && norm.is_finite() && norm > max_norm {
        max_norm / norm
    } else {
        1.0
    }
}

/// Clip in place; returns the pre-clip norm.
pub fn clip_grad_norm(g: &mut Gpt2, max_norm: f32) -> f32 {
    let norm = global_norm(g);
    let s = clip_scale(norm, max_norm);
    if s != 1.0 {
        g.params_mut()
            .into_iter()
            .for_each(|p| p.iter_mut().for_each(|x| *x *= s));
    }
    norm
}

/// One AdamW step (increments `state.t`). `grad_scale` multiplies every gradient first (the
/// clip factor; pass 1.0 for none).
pub fn adam_step(w: &mut Gpt2, g: &Gpt2, state: &mut AdamState, adam: &Adam, grad_scale: f32) {
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

/// Linear warmup then cosine decay to zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    /// Peak learning rate.
    pub base_lr: f32,
    /// Warmup steps.
    pub warmup: u64,
    /// Total steps of the run.
    pub total: u64,
}

impl Schedule {
    /// Learning rate for the 1-based `step`.
    pub fn lr_at(&self, step: u64) -> f32 {
        let step = step.max(1);
        if step <= self.warmup {
            return self.base_lr * (step as f32 / self.warmup.max(1) as f32);
        }
        if step >= self.total {
            return 0.0;
        }
        let decay = self.total.saturating_sub(self.warmup).max(1);
        let progress = (step - self.warmup) as f32 / decay as f32;
        self.base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}
#[cfg(test)]
mod tests;
