use super::*;
use ferrum_wgpu::device::open_for_test;

fn pattern(n: usize) -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(n)
        .copied()
        .collect()
}

fn spec(steps: usize) -> TrainSpec {
    TrainSpec {
        text: pattern(600),
        steps,
        lr: 0.01,
        batch: 4,
        context: 24,
        seed: 3,
    }
}

fn shared() -> Arc<Mutex<Progress>> {
    Arc::new(Mutex::new(Progress::default()))
}

#[test]
fn presets_are_valid_models_and_unknown_names_are_refused() {
    for name in ["tiny", "small", "medium"] {
        let cfg = preset(name).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.vocab, 256);
        let m = fresh_model(name, 1).unwrap();
        assert!(!m.tied && m.num_params() > 0);
    }
    assert!(preset("tiny").unwrap().ctx < preset("medium").unwrap().ctx);
    assert!(preset("huge").is_none());
    assert!(fresh_model("huge", 1)
        .unwrap_err()
        .contains("unknown preset"));
    assert_eq!(
        fresh_model("tiny", 5).unwrap(),
        fresh_model("tiny", 5).unwrap(),
        "seeded"
    );
    assert_ne!(
        fresh_model("tiny", 5).unwrap(),
        fresh_model("tiny", 6).unwrap()
    );
}

#[test]
fn batches_are_shifted_windows_chosen_by_the_seed() {
    let text = pattern(200);
    let (ids, targets) = make_batch(&text, 10, 3, &mut Rng::new(1));
    assert_eq!((ids.len(), targets.len()), (30, 30));
    for w in 0..3 {
        // the target of each position is the next byte of the same window
        assert_eq!(
            &ids[w * 10 + 1..(w + 1) * 10],
            &targets[w * 10..(w + 1) * 10 - 1]
        );
    }
    assert_eq!(
        make_batch(&text, 10, 3, &mut Rng::new(1)),
        (ids.clone(), targets)
    );
    assert_ne!(make_batch(&text, 10, 3, &mut Rng::new(2)).0, ids);
    // the last window may end exactly at the final byte
    let (i, t) = make_batch(b"abcd", 3, 50, &mut Rng::new(9));
    assert!(i.iter().chain(&t).all(|&x| (97..=100).contains(&x)));
}

#[test]
fn train_specs_are_validated_with_the_reason() {
    let cfg = preset("tiny").unwrap();
    assert!(spec(5).validate(&cfg).is_ok());
    let bad = |f: &dyn Fn(&mut TrainSpec)| {
        let mut s = spec(5);
        f(&mut s);
        s.validate(&cfg).unwrap_err()
    };
    assert!(bad(&|s| s.steps = 0).contains("steps"));
    assert!(bad(&|s| s.steps = 200_000).contains("steps"));
    assert!(bad(&|s| s.batch = 0).contains("batch"));
    assert!(bad(&|s| s.batch = 1000).contains("batch"));
    assert!(bad(&|s| s.context = 0).contains("context"));
    assert!(bad(&|s| s.context = 65).contains("context"));
    assert!(bad(&|s| s.text = vec![1; 24]).contains("needs more"));
    assert!(bad(&|s| s.lr = 0.0).contains("lr"));
    assert!(bad(&|s| s.lr = f32::NAN).contains("lr"));
    assert!(bad(&|s| s.lr = 5.0).contains("lr"));
}

#[test]
fn the_loop_reports_progress_follows_the_schedule_and_can_be_stopped() {
    let progress = shared();
    let mut lrs = Vec::new();
    run_training(&spec(40), &progress, &mut |ids, targets, batch, cfg| {
        assert_eq!((ids.len(), targets.len(), batch), (96, 96, 4));
        lrs.push(cfg.adam.lr);
        assert_eq!(cfg.clip, 1.0);
        Ok(StepStats {
            loss: 2.0,
            grad_norm: 1.0,
        })
    })
    .unwrap();
    let p = progress.lock().unwrap().clone();
    assert_eq!((p.step, p.total, p.losses.len()), (40, 40, 40));
    assert!(
        lrs[0] < lrs[1] && lrs[1] < lrs[3] && (lrs[3] - 0.01).abs() < 1e-6,
        "warmup (4 steps) rises to the peak: {lrs:?}"
    );
    assert!(
        lrs[39] < lrs[10] && lrs[10] < lrs[3],
        "cosine decay falls: {lrs:?}"
    );
    // a stop request ends the run cleanly after the current step
    let progress = shared();
    let flag = progress.clone();
    run_training(&spec(50), &progress, &mut |_, _, _, _| {
        flag.lock().unwrap().stop = true;
        Ok(StepStats {
            loss: 1.0,
            grad_norm: 1.0,
        })
    })
    .unwrap();
    assert_eq!(progress.lock().unwrap().step, 1);
    // a diverged loss and an engine error both end the run with a message
    let e = run_training(&spec(5), &shared(), &mut |_, _, _, _| {
        Ok(StepStats {
            loss: f32::NAN,
            grad_norm: 0.0,
        })
    })
    .unwrap_err();
    assert!(e.contains("lower the learning rate"), "{e}");
    let e = run_training(&spec(5), &shared(), &mut |_, _, _, _| {
        Err("device lost".into())
    })
    .unwrap_err();
    assert_eq!(e, "device lost");
}

#[test]
fn generation_respects_limits_and_the_sampler() {
    let opts = GenOpts {
        max_new: 5,
        temperature: 0.0,
        top_k: 0,
        seed: 1,
    };
    let mut fed = Vec::new();
    // logits that always prefer (last token + 1) % 8
    let mut feed = |t: usize| -> Result<Vec<f32>, String> {
        fed.push(t);
        let mut l = vec![0.0; 8];
        l[(t + 1) % 8] = 5.0;
        Ok(l)
    };
    let out = generate_with(&mut feed, 100, &[2, 3], &opts).unwrap();
    assert_eq!(out, vec![4, 5, 6, 7, 0], "greedy follows the pattern");
    assert_eq!(
        fed,
        vec![2, 3, 4, 5, 6, 7],
        "the last token is not fed again"
    );
    // the context window ends generation early
    let mut feed2 = |_: usize| -> Result<Vec<f32>, String> { Ok(vec![1.0, 0.0]) };
    let out = generate_with(
        &mut feed2,
        5,
        &[1, 1],
        &GenOpts {
            max_new: 50,
            ..opts
        },
    )
    .unwrap();
    assert_eq!(out.len(), 3, "5 - 2 prompt tokens");
    assert!(generate_with(&mut feed2, 5, &[], &opts)
        .unwrap_err()
        .contains("empty"));
    assert!(generate_with(&mut feed2, 2, &[1, 1], &opts)
        .unwrap_err()
        .contains("context"));
    let mut failing = |_: usize| -> Result<Vec<f32>, String> { Err("boom".into()) };
    assert_eq!(
        generate_with(&mut failing, 9, &[1], &opts).unwrap_err(),
        "boom"
    );
    // temperature sampling is seeded: same seed, same tokens
    let hot = GenOpts {
        temperature: 1.5,
        top_k: 3,
        max_new: 20,
        ..opts
    };
    let mut flat =
        |_: usize| -> Result<Vec<f32>, String> { Ok((0..8).map(|i| i as f32 * 0.1).collect()) };
    let (a, b) = (
        generate_with(&mut flat, 99, &[1], &hot).unwrap(),
        generate_with(&mut flat, 99, &[1], &hot).unwrap(),
    );
    assert_eq!(a, b);
    assert!(a.iter().all(|&t| t < 8));
}

fn train_on(backend: &dyn Backend, steps: usize) -> (Llama, Vec<f32>) {
    let progress = shared();
    let w = backend
        .train(&fresh_model("tiny", 7).unwrap(), &spec(steps), &progress)
        .unwrap();
    let losses = progress.lock().unwrap().losses.clone();
    (w, losses)
}

#[test]
fn the_cpu_backend_learns_a_repeating_text_and_continues_it() {
    let cpu = CpuBackend::new(2);
    assert_eq!(cpu.kind(), "cpu");
    assert!(cpu.name().contains("2 threads"));
    assert!(CpuBackend::new(0).name().contains("1 threads"));
    let (w, losses) = train_on(&cpu, 60);
    assert_eq!(losses.len(), 60);
    assert!(
        losses[59] < 0.5 * losses[0],
        "{} -> {}",
        losses[0],
        losses[59]
    );
    let prompt = crate::tokenizer::encode("the quick brown ");
    let out = cpu
        .generate(
            &w,
            &prompt,
            &GenOpts {
                max_new: 12,
                temperature: 0.0,
                top_k: 0,
                seed: 1,
            },
        )
        .unwrap();
    let text = crate::tokenizer::decode(&out).unwrap();
    eprintln!("cpu continuation: {text:?}");
    assert!(
        text.starts_with("fox"),
        "a trained model continues the pattern, got {text:?}"
    );
    // training refuses a spec that does not fit
    let mut s = spec(5);
    s.context = 1000;
    assert!(cpu.train(&w, &s, &shared()).is_err());
}

#[test]
fn the_gpu_backend_matches_the_cpu_backend() {
    let Some(g) = open_for_test() else { return };
    let gpu = GpuBackend::new(Arc::new(g));
    assert_eq!(gpu.kind(), "gpu");
    assert!(!gpu.name().is_empty());
    let (wc, lc) = train_on(&CpuBackend::new(2), 30);
    let (wg, lg) = train_on(&gpu, 30);
    let worst = lc
        .iter()
        .zip(&lg)
        .map(|(a, b)| (a - b).abs() / a.max(1.0))
        .fold(0.0f32, f32::max);
    eprintln!("cpu/gpu loss curves differ by at most {worst:.2e}");
    assert!(worst < 1e-3, "{worst}");
    assert!(lg[29] < 0.7 * lg[0]);
    // the same weights decode to the same tokens on either backend
    let prompt = crate::tokenizer::encode("the quick ");
    let opts = GenOpts {
        max_new: 10,
        temperature: 0.0,
        top_k: 0,
        seed: 1,
    };
    let (a, b) = (
        CpuBackend::new(1).generate(&wc, &prompt, &opts).unwrap(),
        gpu.generate(&wc, &prompt, &opts).unwrap(),
    );
    assert_eq!(a, b);
    assert_eq!(gpu.generate(&wg, &prompt, &opts).unwrap().len(), 10);
    let mut s = spec(5);
    s.lr = 0.0;
    assert!(gpu.train(&wc, &s, &shared()).is_err());
    assert!(GpuBackend::open(Some("no-such-adapter")).is_err());
}
