use super::*;

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
    let moments = (&mut state.m, &mut state.v, &mut state.t);
    crate::training::optim::adam_step_model(adam, (w, g), moments, grad_scale);
}

/// The CPU Llama trainer (also the oracle for the GPU trainer).
pub struct LlamaTrainer {
    pub(super) w: Llama,
    pub(super) state: Moments,
    pub(super) threads: usize,
}
