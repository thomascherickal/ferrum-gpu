use super::*;

pub(super) fn cfg() -> Config {
    Config {
        vocab: 11,
        ctx: 8,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 16,
    }
}

/// A tiny learnable corpus: a repeating pattern.
pub(super) fn corpus(n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| [3usize, 7, 1, 9, 4, 4, 2, 8][i % 8])
        .collect()
}

/// Records what the loop asks of an engine, without doing any math.
#[derive(Default)]
pub(super) struct Mock {
    pub(super) calls: Vec<(Vec<usize>, Vec<usize>, usize, f32)>,
    pub(super) done: u64,
}
impl Engine for Mock {
    fn step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        batch: usize,
        c: &StepConfig,
    ) -> Result<StepStats, String> {
        self.calls
            .push((ids.to_vec(), targets.to_vec(), batch, c.adam.lr));
        self.done += 1;
        Ok(StepStats {
            loss: self.done as f32,
            grad_norm: 0.0,
        })
    }
    fn steps(&self) -> u64 {
        self.done
    }
    fn snapshot(&self) -> Result<Snapshot, String> {
        let w = Gpt2::random(cfg(), 1);
        Ok(Snapshot {
            weights: w.clone(),
            m: w.zeros_like(),
            v: w.zeros_like(),
            steps: self.done,
        })
    }
    fn restore(&mut self, s: &Snapshot) -> Result<(), String> {
        self.done = s.steps;
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct Rec {
    pub(super) steps: Vec<u64>,
    pub(super) checkpoints: Vec<(u64, u64, u64)>,
    pub(super) stop_at: Option<u64>,
}
impl Hooks for Rec {
    fn on_step(&mut self, step: u64, _s: &StepStats) -> bool {
        self.steps.push(step);
        self.stop_at != Some(step)
    }
    fn on_checkpoint(&mut self, cp: &Checkpoint) -> Result<(), String> {
        self.checkpoints
            .push((cp.snapshot.steps, cp.epoch, cp.step_in_epoch));
        Ok(())
    }
}

pub(super) fn tied(mut w: Gpt2) -> Gpt2 {
    w.tie_head();
    w
}
