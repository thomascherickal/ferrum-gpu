use super::*;

#[test]
fn generate_with_stops_at_once_when_the_request_is_cancelled() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let opts = GenOpts {
        max_new: 50,
        temperature: 0.0,
        top_k: 0,
        seed: 1,
    };
    let flag = std::sync::Arc::new(AtomicBool::new(false));
    let _scope = crate::cancel::install(flag.clone());
    let mut fed = 0usize;
    let mut feed = |t: usize| -> Result<Vec<f32>, String> {
        fed += 1;
        if fed == 3 {
            flag.store(true, Ordering::Relaxed); // the client leaves while decoding
        }
        let mut l = vec![0.0; 8];
        l[(t + 1) % 8] = 5.0;
        Ok(l)
    };
    let e = generate_with(&mut feed, 100, &[1, 2], &opts).unwrap_err();
    assert_eq!(e, crate::cancel::MESSAGE);
    assert!(fed <= 4, "stopped within a token of the flag: fed {fed}");
}

#[test]
fn fresh_model_with_ties_the_head_and_the_default_does_not() {
    assert!(fresh_model_with("tiny", 1, true).unwrap().tied);
    assert!(!fresh_model_with("tiny", 1, false).unwrap().tied);
    assert!(fresh_model_with("huge", 1, true).is_err());
}

#[test]
fn a_gpu_decoder_with_stored_weights_stays_close_to_the_f32_one() {
    use ferrum_wgpu::ops_quant::QKind;
    let Some(g) = open_for_test() else { return };
    let g = Arc::new(g);
    let (w, _) = train_on(&CpuBackend::new(2), 5);
    let prompt = crate::tokenizer::encode("the quick ");
    let logits = |weights| {
        let b = GpuBackend::new(g.clone()).with_weights(weights);
        let mut feed = b.llama_decoder(&w).unwrap();
        let mut last = Vec::new();
        for &t in &prompt {
            last = feed(t).unwrap();
        }
        last
    };
    let exact = logits(None);
    // f16 keeps 11 bits, so every logit stays within a small relative error of the f32 one
    let f16 = logits(Some(QKind::F16));
    let e = ferrum_wgpu::tolerance::max_rel_err(&f16, &exact);
    eprintln!("f16 weights vs f32 weights: max relative logit error {e:.2e}");
    assert!(e < 5e-3, "{e}");
    // generation runs end to end with each stored precision
    let opts = GenOpts {
        max_new: 4,
        temperature: 0.0,
        top_k: 0,
        seed: 1,
    };
    for q in [QKind::F16, QKind::Q8_0, QKind::Q4_0] {
        let b = GpuBackend::new(g.clone()).with_weights(Some(q));
        assert_eq!(b.generate(&w, &prompt, &opts).unwrap().len(), 4, "{q:?}");
    }
}

#[test]
fn the_gpu_gpt2_decoder_matches_the_cpu_one_token_by_token() {
    use ferrum_ref::gpt2::{Config as GptConfig, Gpt2};
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::init(
        GptConfig {
            vocab: 20,
            ctx: 8,
            dim: 32,
            heads: 4,
            layers: 1,
            hidden: 64,
        },
        5,
    );
    let (cpu, gpu) = (CpuBackend::new(1), GpuBackend::new(Arc::new(g)));
    let (mut fc, mut fg) = (cpu.gpt2_decoder(&w).unwrap(), gpu.gpt2_decoder(&w).unwrap());
    for t in [1usize, 5, 3, 9] {
        let (a, b) = (fc(t).unwrap(), fg(t).unwrap());
        let worst = ferrum_wgpu::tolerance::max_rel_err(&b, &a);
        assert!(
            worst < ferrum_wgpu::tolerance::FORWARD,
            "token {t}: {worst}"
        );
    }
}
