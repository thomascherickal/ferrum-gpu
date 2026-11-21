use super::*;

impl Hooks for Run<'_> {
    fn on_step(&mut self, step: u64, s: &StepStats) -> bool {
        let el = self.started.elapsed().as_secs_f64();
        let tps = (step as usize * self.tokens_per_step) as f64 / el.max(1e-9);
        let (epoch, lr) = ((step - 1) / self.steps_per_epoch, self.lr_at(step));
        if let Some(f) = self.csv.as_mut() {
            let row = format!(
                "{step},{epoch},{},{lr},{},{tps:.1},{el:.3}\n",
                s.loss, s.grad_norm
            );
            if let Err(e) = f.write_all(row.as_bytes()) {
                self.error = Some(format!("--metrics-csv: {e}"));
                return false;
            }
        }
        if self.verbose && (step.is_multiple_of(10) || step == self.total) {
            let eta = el / step as f64 * (self.total - step) as f64;
            self.note(format!(
                "step {step}/{} loss {:.4} grad_norm {:.3} tokens/s {tps:.0} elapsed {el:.1}s eta {eta:.0}s",
                self.total, s.loss, s.grad_norm
            ));
        }
        if !s.loss.is_finite() {
            self.error = Some(format!(
                "the loss became {} at step {step}; lower --lr or add --clip",
                s.loss
            ));
            return false;
        }
        if self.max_seconds.is_some_and(|m| el >= m) {
            self.reason = Some(format!("--max-seconds reached after step {step}"));
            return false;
        }
        if self.stop_file.exists() {
            self.reason = Some(format!(
                "{} found; stopped after step {step}",
                self.stop_file.display()
            ));
            return false;
        }
        if crate::cancel::requested() {
            self.reason = Some(format!("cancelled; stopped after step {step}"));
            return false;
        }
        self.reason.is_none()
    }

    fn on_checkpoint(&mut self, cp: &Checkpoint) -> Result<(), String> {
        if let Some(p) = &self.checkpoint {
            atomic_write(p, &to_bytes(cp)).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        let w = &cp.snapshot.weights;
        if let Some((tokens, patience)) = self.val.as_ref().map(|v| (&v.tokens, v.patience)) {
            let (loss, _) = mean_loss(w, tokens, w.cfg.ctx)?;
            let better = self.best.as_ref().is_none_or(|b| loss < b.0);
            let epoch = cp.snapshot.steps / self.steps_per_epoch.max(1);
            self.note(format!(
                "epoch {epoch} val loss {loss:.4}{}",
                if better { " (best)" } else { "" }
            ));
            if better {
                self.best = Some((loss, w.clone()));
                self.bad_epochs = 0;
            } else {
                self.bad_epochs += 1;
                if patience > 0 && self.bad_epochs >= patience {
                    self.reason = Some(format!("no validation improvement for {patience} epochs"));
                }
            }
        }
        self.last = Some(w.clone());
        Ok(())
    }
}
