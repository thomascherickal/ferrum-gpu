use super::*;
use crate::gpt2::{Config, Gpt2};

fn cfg() -> Config {
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
fn corpus(n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| [3usize, 7, 1, 9, 4, 4, 2, 8][i % 8])
        .collect()
}

/// Records what the loop asks of an engine, without doing any math.
#[derive(Default)]
struct Mock {
    calls: Vec<(Vec<usize>, Vec<usize>, usize, f32)>,
    done: u64,
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
struct Rec {
    steps: Vec<u64>,
    checkpoints: Vec<(u64, u64, u64)>,
    stop_at: Option<u64>,
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

fn tied(mut w: Gpt2) -> Gpt2 {
    w.tie_head();
    w
}

/// An engine whose every step fails.
struct Failing;
impl Engine for Failing {
    fn step(
        &mut self,
        _: &[usize],
        _: &[usize],
        _: usize,
        _: &StepConfig,
    ) -> Result<StepStats, String> {
        Err("device exploded".into())
    }
    fn steps(&self) -> u64 {
        0
    }
    fn snapshot(&self) -> Result<Snapshot, String> {
        Err("no state".into())
    }
    fn restore(&mut self, _: &Snapshot) -> Result<(), String> {
        Ok(())
    }
}

/// A source that computes its tokens on demand, like a file read window by window.
struct Lazy(usize);

impl TokenSource for Lazy {
    fn len(&self) -> usize {
        self.0
    }
    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        if start + len > self.0 {
            return Err("past the end".into());
        }
        Ok((start..start + len).map(|i| (i * 7) % 50).collect())
    }
}

mod a_prefix_hides_the_held_out_tail;
mod a_tied_trainer_keeps_the_head_exactly_the_embedding_transposed_and_still_learns;
mod checkpoints_follow_the_cadence_and_always_come_at_the_end;
mod qat_trains_the_masters_through_the_snapped_weights;
mod resume_refuses_a_changed_run_shape;
mod windows_start_every_stride_while_a_target_remains;
