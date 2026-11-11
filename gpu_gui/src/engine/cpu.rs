//! The CPU backend: `ferrum_ref::llama`, the oracle the GPU is checked against.

use super::{generate_with, run_training, Backend, GenOpts, Progress, TrainSpec};
use ferrum_ref::llama::{Cache, Llama, LlamaTrainer};
use std::sync::{Arc, Mutex};

/// Runs everything on the CPU.
pub struct CpuBackend {
    threads: usize,
}

impl CpuBackend {
    /// A CPU backend using `threads` matmul threads for training (at least one).
    pub fn new(threads: usize) -> Self {
        CpuBackend {
            threads: threads.max(1),
        }
    }
}

impl Backend for CpuBackend {
    fn kind(&self) -> &'static str {
        "cpu"
    }

    fn name(&self) -> String {
        format!("CPU ({} threads)", self.threads)
    }

    fn train(
        &self,
        model: &Llama,
        spec: &TrainSpec,
        progress: &Arc<Mutex<Progress>>,
    ) -> Result<Llama, String> {
        spec.validate(&model.cfg)?;
        let mut tr = LlamaTrainer::new(model.clone(), self.threads);
        run_training(spec, progress, &mut |ids, targets, batch, cfg| {
            tr.step(ids, targets, batch, cfg)
        })?;
        Ok(tr.weights().clone())
    }

    fn generate(
        &self,
        model: &Llama,
        prompt: &[usize],
        opts: &GenOpts,
    ) -> Result<Vec<usize>, String> {
        let mut cache = Cache::new(model.cfg.layers);
        generate_with(
            &mut |t| model.feed(&mut cache, t),
            model.cfg.ctx,
            prompt,
            opts,
        )
    }
}
