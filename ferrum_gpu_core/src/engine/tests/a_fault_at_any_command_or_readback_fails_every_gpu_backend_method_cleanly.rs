use super::*;
use ferrum_ref::gpt2::{Config as GptConfig, Gpt2};
use ferrum_wgpu::GpuError;

fn failing<T>(r: Result<T, String>) -> Result<(), GpuError> {
    r.map(|_| ()).map_err(GpuError)
}

/// A GPT-2 whose widths are multiples of 32, so quantisation-aware training applies.
fn gpt2() -> Gpt2 {
    let c = GptConfig {
        vocab: 16,
        ctx: 8,
        dim: 32,
        heads: 2,
        layers: 1,
        hidden: 64,
    };
    Gpt2::init(c, 5)
}

#[test]
fn a_fault_at_any_command_fails_every_gpu_backend_method_cleanly() {
    let Some(g) = open_for_test() else { return };
    let g = Arc::new(g);
    let (b, w, llama) = (
        GpuBackend::new(g.clone()),
        gpt2(),
        fresh_model("tiny", 7).unwrap(),
    );
    let tie = EngineOptions {
        tie: true,
        dropout: 0.0,
        seed: 1,
    };
    g.sweep_faults(|| failing(b.gpt2_engine(&w, true)));
    g.sweep_faults(|| failing(b.gpt2_engine_with(&w, true, &tie)));
    g.sweep_faults(|| failing(b.gpt2_decoder(&w).and_then(|mut feed| feed(3))));
    g.sweep_faults(|| failing(b.llama_decoder(&llama).and_then(|mut feed| feed(3))));
    let opts = GenOpts {
        max_new: 2,
        temperature: 0.0,
        top_k: 0,
        seed: 1,
    };
    g.sweep_faults(|| failing(b.generate(&llama, &[1, 2], &opts)));
    g.sweep_faults(|| failing(b.train(&llama, &spec(1), &shared())));
}

/// Run `f` failing allocation 0, 1, 2, ... until it succeeds; each failure must be the injected one.
fn sweep_allocs(g: &ferrum_wgpu::Gpu, mut f: impl FnMut() -> Result<(), String>) {
    let finished = (0..10_000).find(|&k| {
        g.fail_allocs_after(k);
        match f() {
            Ok(()) => true,
            Err(e) => {
                assert!(e.contains("injected fault"), "allocation {k}: {e}");
                false
            }
        }
    });
    g.fail_allocs_after(usize::MAX);
    assert!(finished.is_some());
}

#[test]
fn a_failed_allocation_fails_every_gpu_backend_method_cleanly() {
    let Some(g) = open_for_test() else { return };
    let g = Arc::new(g);
    let (b, w, llama) = (
        GpuBackend::new(g.clone()),
        gpt2(),
        fresh_model("tiny", 7).unwrap(),
    );
    let tie = EngineOptions {
        tie: true,
        dropout: 0.0,
        seed: 1,
    };
    let opts = GenOpts {
        max_new: 2,
        temperature: 0.0,
        top_k: 0,
        seed: 1,
    };
    sweep_allocs(&g, || b.gpt2_engine(&w, true).map(|_| ()));
    sweep_allocs(&g, || b.gpt2_engine_with(&w, true, &tie).map(|_| ()));
    sweep_allocs(&g, || b.gpt2_decoder(&w).and_then(|mut f| f(3)).map(|_| ()));
    sweep_allocs(&g, || {
        b.llama_decoder(&llama).and_then(|mut f| f(3)).map(|_| ())
    });
    sweep_allocs(&g, || b.generate(&llama, &[1, 2], &opts).map(|_| ()));
    sweep_allocs(&g, || b.train(&llama, &spec(1), &shared()).map(|_| ()));
}
