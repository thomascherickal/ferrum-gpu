use super::*;

/// Parallel tensor lists for one AdamW step: weights, gradients, first and second moments.
pub(crate) struct AdamLists<'a> {
    pub(crate) w: Vec<&'a mut Vec<f32>>,
    pub(crate) g: Vec<&'a Vec<f32>>,
    pub(crate) m: Vec<&'a mut Vec<f32>>,
    pub(crate) v: Vec<&'a mut Vec<f32>>,
}

/// A model whose parameters an optimizer walks: the tensors in a fixed order and which of them
/// take weight decay. Implemented by both model families so one AdamW step serves both.
pub(crate) trait ParamSet {
    fn tensors(&self) -> Vec<&Vec<f32>>;
    fn tensors_mut(&mut self) -> Vec<&mut Vec<f32>>;
    fn takes_decay(&self, i: usize) -> bool;
}

impl ParamSet for crate::gpt2::Gpt2 {
    fn tensors(&self) -> Vec<&Vec<f32>> {
        self.params()
    }
    fn tensors_mut(&mut self) -> Vec<&mut Vec<f32>> {
        self.params_mut()
    }
    fn takes_decay(&self, i: usize) -> bool {
        self.decays(i)
    }
}

impl ParamSet for crate::llama::Llama {
    fn tensors(&self) -> Vec<&Vec<f32>> {
        self.params()
    }
    fn tensors_mut(&mut self) -> Vec<&mut Vec<f32>> {
        self.params_mut()
    }
    fn takes_decay(&self, i: usize) -> bool {
        self.decays(i)
    }
}

/// One AdamW step of model `w` with gradients `g` and moments `(m, v)` at step `*t + 1`
/// (incremented first); `grad_scale` multiplies every gradient first.
pub(crate) fn adam_step_model<M: ParamSet>(
    adam: &Adam,
    (w, g): (&mut M, &M),
    (m, v, t): (&mut M, &mut M, &mut u64),
    grad_scale: f32,
) {
    *t += 1;
    let decays: Vec<bool> = (0..w.tensors().len()).map(|i| w.takes_decay(i)).collect();
    let lists = AdamLists {
        w: w.tensors_mut(),
        g: g.tensors(),
        m: m.tensors_mut(),
        v: v.tensors_mut(),
    };
    adam.step_tensors(*t, grad_scale, &decays, lists);
}

impl Adam {
    /// One AdamW step over every tensor at step `t`, the one implementation both models use.
    /// `grad_scale` multiplies every gradient first (the clip factor, 1.0 for none); `decays[i]` says
    /// whether tensor `i` takes weight decay.
    pub(crate) fn step_tensors(
        &self,
        t: u64,
        grad_scale: f32,
        decays: &[bool],
        lists: AdamLists<'_>,
    ) {
        let (bc1, bc2) = self.bias_corrections(t);
        let tensors = lists.w.into_iter().zip(lists.g).zip(lists.m).zip(lists.v);
        for (i, (((p, gp), mp), vp)) in tensors.enumerate() {
            let wd = if decays[i] { self.weight_decay } else { 0.0 };
            for j in 0..p.len() {
                let gr = gp[j] * grad_scale;
                mp[j] = self.beta1 * mp[j] + (1.0 - self.beta1) * gr;
                vp[j] = self.beta2 * vp[j] + (1.0 - self.beta2) * gr * gr;
                let (mh, vh) = (mp[j] / bc1, vp[j] / bc2);
                let mut update = self.lr * mh / (vh.sqrt() + self.eps);
                if wd != 0.0 {
                    update += self.lr * wd * p[j];
                }
                p[j] -= update;
            }
        }
    }
}
