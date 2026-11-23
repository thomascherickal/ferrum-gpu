use super::*;

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
