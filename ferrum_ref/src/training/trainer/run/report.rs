use super::*;

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
pub fn train<T: TokenSource + ?Sized>(
    engine: &mut dyn Engine,
    tokens: &T,
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
    let starts = window_starts(TokenSource::len(tokens), cfg.context, stride);
    if starts.is_empty() {
        return Err(format!(
            "{} tokens are too few for windows of {} (+1)",
            TokenSource::len(tokens),
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
            let win = tokens.window(starts[w], cfg.context + 1)?;
            ids.extend_from_slice(&win[..cfg.context]);
            targets.extend_from_slice(&win[1..]);
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
