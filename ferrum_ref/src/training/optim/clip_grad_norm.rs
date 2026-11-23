use super::*;

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
    let decays: Vec<bool> = (0..w.params().len()).map(|i| w.decays(i)).collect();
    let lists = AdamLists {
        w: w.params_mut(),
        g: g.params(),
        m: state.m.params_mut(),
        v: state.v.params_mut(),
    };
    adam.step_tensors(state.t, grad_scale, &decays, lists);
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
