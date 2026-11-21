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

/// Feed `prompt`, then sample `opts.max_new` tokens (stopping when the context is full), using
/// `feed` to turn a token into the next-token logits. Returns only the new tokens.
pub fn generate_with(
    feed: &mut dyn FnMut(usize) -> Result<Vec<f32>, String>,
    ctx: usize,
    prompt: &[usize],
    opts: &GenOpts,
) -> Result<Vec<usize>, String> {
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    if prompt.len() >= ctx {
        return Err(format!(
            "the prompt has {} tokens but the context is {ctx}",
            prompt.len()
        ));
    }
    let params = Params {
        temperature: opts.temperature,
        top_k: opts.top_k,
        ..Params::default()
    };
    let mut rng = Rng::new(opts.seed);
    let mut logits = Vec::new();
    for &t in prompt {
        crate::cancel::check()?;
        logits = feed(t)?;
    }
    let mut history = prompt.to_vec();
    let mut out = Vec::new();
    while out.len() < opts.max_new && history.len() < ctx {
        crate::cancel::check()?;
        let next = sample(&logits, &params, &history, &mut rng);
        out.push(next);
        history.push(next);
        if history.len() < ctx && out.len() < opts.max_new {
            logits = feed(next)?;
        }
    }
    Ok(out)
}

/// Training extras a backend may or may not support (see [`Backend::gpt2_engine_with`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineOptions {
    /// Share the token embedding and the output head.
    pub tie: bool,
    /// FFN-hidden dropout rate (`0.0` is off).
    pub dropout: f32,
    /// The run's seed (dropout masks are a function of it and the step).
    pub seed: u64,
}

/// A one-token-at-a-time GPT-2 decoder: feed a token id, get the next-token logits.
pub type Gpt2Feed<'a> = Box<dyn FnMut(usize) -> Result<Vec<f32>, String> + 'a>;
