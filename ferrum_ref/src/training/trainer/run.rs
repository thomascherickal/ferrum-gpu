//! The backend-independent training loop: run configuration, checkpoints, hooks, `train`.

use super::engine::{Engine, Snapshot, StepConfig, StepStats};
use crate::optim::{Adam, Schedule};
use crate::rng::Rng;

/// How a run walks the corpus and steps the optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    /// Tokens per training window (the model sees `context` inputs and predicts the next token of each).
    pub context: usize,
    /// Windows per step (the last step of an epoch may hold fewer).
    pub batch: usize,
    /// Tokens between window starts; `0` means `context` (non-overlapping).
    pub stride: usize,
    /// Epochs to run (ignored when `max_steps > 0`).
    pub epochs: usize,
    /// Total optimizer steps for the run; `0` means `epochs` full epochs.
    pub max_steps: u64,
    /// Peak learning rate.
    pub lr: f32,
    /// AdamW weight decay.
    pub weight_decay: f32,
    /// Global-norm clip cap (`0.0` = off).
    pub clip: f32,
    /// Linear warmup steps (only with `cosine`).
    pub warmup: u64,
    /// Warmup then cosine decay to zero over the whole run; otherwise a fixed `lr`.
    pub cosine: bool,
    /// Seed of the window order.
    pub seed: u64,
    /// Offer a checkpoint every this many steps (`0` = only at the end).
    pub checkpoint_every: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            context: 16,
            batch: 16,
            stride: 0,
            epochs: 1,
            max_steps: 0,
            lr: 0.01,
            weight_decay: 0.0,
            clip: 0.0,
            warmup: 0,
            cosine: false,
            seed: 1337,
            checkpoint_every: 0,
        }
    }
}

/// The run parameters a checkpoint must be resumed with (changing any changes the window order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunShape {
    /// Window length.
    pub context: u64,
    /// Windows per step.
    pub batch: u64,
    /// Window stride (resolved, never 0).
    pub stride: u64,
    /// Windows in the corpus.
    pub windows: u64,
    /// Order seed.
    pub seed: u64,
}

/// A resumable training state: the engine snapshot plus where the loop was.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    /// Engine state.
    pub snapshot: Snapshot,
    /// Epoch in progress (0-based).
    pub epoch: u64,
    /// Steps already done within that epoch.
    pub step_in_epoch: u64,
    /// The shape of the run that produced it.
    pub shape: RunShape,
    /// Identity of the corpus, if the caller recorded one.
    pub corpus_id: Option<String>,
}

/// Callbacks for a run. All have no-op defaults.
pub trait Hooks {
    /// After every step; return `false` to stop (a final checkpoint is still offered).
    fn on_step(&mut self, _step: u64, _stats: &StepStats) -> bool {
        true
    }
    /// A checkpoint is due (cadence, stop, or end of run).
    fn on_checkpoint(&mut self, _cp: &Checkpoint) -> Result<(), String> {
        Ok(())
    }
}

/// Hooks that do nothing.
pub struct NoHooks;
impl Hooks for NoHooks {}

/// What a run reports.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// Steps run by this call.
    pub steps_run: u64,
    /// Loss of every step this call ran.
    pub losses: Vec<f32>,
    /// Engine step counter at the end.
    pub final_step: u64,
    /// Stopped early by the hook (rather than finishing the budget).
    pub stopped: bool,
}

/// Window start offsets: every `stride` tokens while a full `context + 1` tokens remain.
pub fn window_starts(n_tokens: usize, context: usize, stride: usize) -> Vec<usize> {
    let stride = if stride == 0 { context } else { stride };
    if context == 0 || stride == 0 || n_tokens < context + 1 {
        return Vec::new();
    }
    (0..=n_tokens - context - 1).step_by(stride).collect()
}

/// The visiting order of `n` windows in epoch `epoch`: a seeded shuffle that depends only on
/// `(seed, epoch)`, so a resumed run reproduces it without any saved generator state.
pub fn epoch_order(n: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    Rng::new(seed ^ epoch.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15)).shuffle(&mut order);
    order
}

/// Steps in one epoch of `n` windows at `batch` windows per step.
pub fn steps_per_epoch(n: usize, batch: usize) -> u64 {
    n.div_ceil(batch.max(1)) as u64
}

/// Run training. Continues from `engine.steps()` (0 for a fresh engine, the checkpoint's step
/// after [`Engine::restore`]); pass the checkpoint's `shape` as `expect` to refuse a resume
/// whose window order would differ.
pub fn train(
    engine: &mut dyn Engine,
    tokens: &[usize],
    cfg: &TrainConfig,
    expect: Option<&RunShape>,
    corpus_id: Option<&str>,
    hooks: &mut dyn Hooks,
) -> Result<Report, String> {
    if cfg.context == 0 || cfg.batch == 0 {
        return Err("context and batch must be positive".into());
    }
    let stride = if cfg.stride == 0 {
        cfg.context
    } else {
        cfg.stride
    };
    let starts = window_starts(tokens.len(), cfg.context, stride);
    if starts.is_empty() {
        return Err(format!(
            "{} tokens are too few for windows of {} (+1)",
            tokens.len(),
            cfg.context
        ));
    }
    let shape = RunShape {
        context: cfg.context as u64,
        batch: cfg.batch as u64,
        stride: stride as u64,
        windows: starts.len() as u64,
        seed: cfg.seed,
    };
    if let Some(e) = expect {
        if *e != shape {
            return Err(format!(
                "cannot resume: the run shape changed (checkpoint {e:?}, now {shape:?})"
            ));
        }
    }
    let spe = steps_per_epoch(starts.len(), cfg.batch);
    let total = if cfg.max_steps > 0 {
        cfg.max_steps
    } else {
        cfg.epochs as u64 * spe
    };
    let sched = Schedule {
        base_lr: cfg.lr,
        warmup: cfg.warmup,
        total,
    };
    let make_checkpoint = |engine: &dyn Engine, done: u64| -> Result<Checkpoint, String> {
        Ok(Checkpoint {
            snapshot: engine.snapshot()?,
            epoch: done / spe,
            step_in_epoch: done % spe,
            shape: shape.clone(),
            corpus_id: corpus_id.map(String::from),
        })
    };
    let mut done = engine.steps();
    let (mut losses, mut stopped) = (Vec::new(), false);
    let mut cached: Option<(u64, Vec<usize>)> = None;
    while done < total {
        let (epoch, k) = (done / spe, (done % spe) as usize);
        if cached.as_ref().is_none_or(|c| c.0 != epoch) {
            cached = Some((epoch, epoch_order(starts.len(), cfg.seed, epoch)));
        }
        let order = &cached.as_ref().expect("just set").1;
        let picked = &order[k * cfg.batch..((k + 1) * cfg.batch).min(order.len())];
        let (mut ids, mut targets) = (
            Vec::with_capacity(picked.len() * cfg.context),
            Vec::with_capacity(picked.len() * cfg.context),
        );
        for &w in picked {
            let s = starts[w];
            ids.extend_from_slice(&tokens[s..s + cfg.context]);
            targets.extend_from_slice(&tokens[s + 1..s + cfg.context + 1]);
        }
        let lr = if cfg.cosine {
            sched.lr_at(done + 1)
        } else {
            cfg.lr
        };
        let adam = Adam {
            weight_decay: cfg.weight_decay,
            ..Adam::new(lr)
        };
        let stats = engine.step(
            &ids,
            &targets,
            picked.len(),
            &StepConfig {
                adam,
                clip: cfg.clip,
            },
        )?;
        done += 1;
        losses.push(stats.loss);
        if !hooks.on_step(done, &stats) {
            stopped = true;
            break;
        }
        if cfg.checkpoint_every > 0 && done.is_multiple_of(cfg.checkpoint_every) && done < total {
            hooks.on_checkpoint(&make_checkpoint(engine, done)?)?;
        }
    }
    hooks.on_checkpoint(&make_checkpoint(engine, done)?)?;
    Ok(Report {
        steps_run: losses.len() as u64,
        losses,
        final_step: done,
        stopped,
    })
}
