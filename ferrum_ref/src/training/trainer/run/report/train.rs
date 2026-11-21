use super::*;

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
