use super::*;

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
