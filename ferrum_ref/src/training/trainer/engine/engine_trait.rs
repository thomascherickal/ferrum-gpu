use super::*;

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
