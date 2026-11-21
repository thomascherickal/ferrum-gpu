use super::*;

/// The training loop both backends share: warmup + cosine learning rate, global-norm clip 1.0,
/// AdamW, progress after every step, early stop. `step` runs one optimizer step.
pub fn run_training(
    spec: &TrainSpec,
    progress: &Arc<Mutex<Progress>>,
    step: &mut StepFn,
) -> Result<(), String> {
    let sched = Schedule {
        base_lr: spec.lr,
        warmup: (spec.steps as u64 / 10).clamp(1, 20),
        total: spec.steps as u64,
    };
    let mut rng = Rng::new(spec.seed);
    lock(progress).total = spec.steps;
    for i in 1..=spec.steps {
        let mut p = lock(progress);
        if crate::cancel::requested() {
            p.stop = true;
        }
        if p.stop {
            break;
        }
        drop(p);
        let (ids, targets) = make_batch(&spec.text, spec.context, spec.batch, &mut rng);
        let cfg = StepConfig {
            adam: Adam {
                weight_decay: 0.01,
                ..Adam::new(sched.lr_at(i as u64))
            },
            clip: 1.0,
        };
        let stats = step(&ids, &targets, spec.batch, &cfg)?;
        if !stats.loss.is_finite() {
            return Err(format!(
                "the loss became {} at step {i}; lower the learning rate",
                stats.loss
            ));
        }
        let mut p = lock(progress);
        p.step = i;
        p.losses.push(stats.loss);
    }
    Ok(())
}

/// Sampling controls for a generation request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenOpts {
    /// Tokens to add after the prompt.
    pub max_new: usize,
    /// `0.0` is greedy.
    pub temperature: f32,
    /// `0` is off.
    pub top_k: usize,
    /// Seed of the sampler.
    pub seed: u64,
}
