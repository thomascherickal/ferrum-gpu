use super::*;

impl<'g> GpuLlamaTrainer<'g> {
    /// Copy the weights back to the host.
    pub fn weights_host(&self) -> Result<Llama, GpuError> {
        self.host(&self.params)
    }

    /// Copy the gradients of the last forward/backward back to the host.
    pub fn grads_host(&self) -> Result<Llama, GpuError> {
        self.host(&self.grads)
    }

    /// Copy the Adam moments back to the host as `(m, v)`.
    pub fn moments_host(&self) -> Result<(Llama, Llama), GpuError> {
        Ok((self.host(&self.m)?, self.host(&self.v)?))
    }

    /// Copy the full training state to the host.
    pub fn snapshot(&self) -> Result<LlamaSnapshot, GpuError> {
        let (m, v) = self.moments_host()?;
        Ok(LlamaSnapshot {
            weights: self.weights_host()?,
            m,
            v,
            steps: self.t,
        })
    }

    /// Replace weights, Adam moments and the step counter (resume from a snapshot).
    pub fn restore(&mut self, s: &LlamaSnapshot) -> Result<(), GpuError> {
        let shape = |m: &Llama| {
            m.cfg == self.cfg
                && m.params()
                    .iter()
                    .zip(&self.params)
                    .all(|(p, b)| p.len() == b.len())
        };
        if !(shape(&s.weights) && shape(&s.m) && shape(&s.v)) {
            return Err(err("restore: the model shape differs from the trainer's"));
        }
        for (set, src) in [(&self.params, &s.weights), (&self.m, &s.m), (&self.v, &s.v)] {
            for (b, p) in set.iter().zip(src.params()) {
                if !p.is_empty() {
                    self.gpu.copy_from_host(b, p)?;
                }
            }
        }
        self.t = s.steps;
        Ok(())
    }
}
