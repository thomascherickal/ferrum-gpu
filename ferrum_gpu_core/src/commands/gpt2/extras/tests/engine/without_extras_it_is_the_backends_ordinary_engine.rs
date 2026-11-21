use super::*;

#[test]
fn without_extras_it_is_the_backends_ordinary_engine() {
    // identical to asking the backend directly
    let app = cpu();
    let mut direct = app.backend.gpt2_engine(&model(), false).unwrap();
    let ids: Vec<usize> = (0..16).map(|i| (i * 3) % 11).collect();
    let targets: Vec<usize> = ids.iter().map(|i| (i + 1) % 11).collect();
    let cfg = StepConfig {
        adam: Adam::new(0.02),
        clip: 1.0,
    };
    let want: Vec<f32> = (0..3)
        .map(|_| direct.step(&ids, &targets, 2, &cfg).unwrap().loss)
        .collect();
    assert_eq!(losses(&PLAIN, 1), want);
}

#[test]
fn dropout_changes_the_run_and_the_seed_fixes_it() {
    let d = Extras {
        tie: false,
        dropout: 0.3,
    };
    assert_eq!(losses(&d, 5), losses(&d, 5));
    assert_ne!(losses(&d, 5), losses(&d, 6));
    assert_ne!(losses(&d, 5), losses(&PLAIN, 5));
}

#[test]
fn a_tied_head_changes_the_run() {
    let t = Extras {
        tie: true,
        dropout: 0.0,
    };
    assert_ne!(losses(&t, 1), losses(&PLAIN, 1));
}

#[test]
fn the_gpu_backend_has_no_such_engine() {
    // the trait's default: a backend that does not override gpt2_engine_with refuses by name
    struct NoExtras;
    impl crate::engine::Backend for NoExtras {
        fn kind(&self) -> &'static str {
            "gpu"
        }
        fn name(&self) -> String {
            "pretend GPU".into()
        }
        fn train(
            &self,
            _: &ferrum_ref::llama::Llama,
            _: &crate::engine::TrainSpec,
            _: &std::sync::Arc<std::sync::Mutex<crate::engine::Progress>>,
        ) -> Result<ferrum_ref::llama::Llama, String> {
            Err("unused".into())
        }
        fn generate(
            &self,
            _: &ferrum_ref::llama::Llama,
            _: &[usize],
            _: &crate::engine::GenOpts,
        ) -> Result<Vec<usize>, String> {
            Err("unused".into())
        }
    }
    let e = crate::engine::Backend::gpt2_engine_with(
        &NoExtras,
        &model(),
        false,
        &crate::engine::EngineOptions {
            tie: true,
            dropout: 0.0,
            seed: 1,
        },
    )
    .err()
    .unwrap();
    assert!(e.contains("not available on pretend GPU"), "{e}");
}
