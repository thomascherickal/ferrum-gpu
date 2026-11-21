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
