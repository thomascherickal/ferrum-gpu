use super::*;

/// Parallel tensor lists for one AdamW step: weights, gradients, first and second moments.
pub(crate) struct AdamLists<'a> {
    pub(crate) w: Vec<&'a mut Vec<f32>>,
    pub(crate) g: Vec<&'a Vec<f32>>,
    pub(crate) m: Vec<&'a mut Vec<f32>>,
    pub(crate) v: Vec<&'a mut Vec<f32>>,
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
