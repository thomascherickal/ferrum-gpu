use crate::cmds::gpt2::extras::{engine, Extras};
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::optim::Adam;
use ferrum_ref::trainer::StepConfig;

fn cpu() -> App {
    App::new(DeviceChoice::Cpu, None, 1).unwrap()
}

fn model() -> Gpt2 {
    Gpt2::init(
        Config {
            vocab: 11,
            ctx: 8,
            dim: 8,
            heads: 2,
            layers: 1,
            hidden: 16,
        },
        3,
    )
}

/// Three steps on a fixed batch; the losses.
fn losses(x: &Extras, seed: u64) -> Vec<f32> {
    let app = cpu();
    let mut e = engine(&app, &model(), false, x, seed).unwrap();
    let ids: Vec<usize> = (0..16).map(|i| (i * 3) % 11).collect();
    let targets: Vec<usize> = ids.iter().map(|i| (i + 1) % 11).collect();
    let cfg = StepConfig {
        adam: Adam::new(0.02),
        clip: 1.0,
    };
    (0..3)
        .map(|_| e.step(&ids, &targets, 2, &cfg).unwrap().loss)
        .collect()
}

const PLAIN: Extras = Extras {
    tie: false,
    dropout: 0.0,
};

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
    impl ferrum_gpu_gui::engine::Backend for NoExtras {
        fn kind(&self) -> &'static str {
            "gpu"
        }
        fn name(&self) -> String {
            "pretend GPU".into()
        }
        fn train(
            &self,
            _: &ferrum_ref::llama::Llama,
            _: &ferrum_gpu_gui::engine::TrainSpec,
            _: &std::sync::Arc<std::sync::Mutex<ferrum_gpu_gui::engine::Progress>>,
        ) -> Result<ferrum_ref::llama::Llama, String> {
            Err("unused".into())
        }
        fn generate(
            &self,
            _: &ferrum_ref::llama::Llama,
            _: &[usize],
            _: &ferrum_gpu_gui::engine::GenOpts,
        ) -> Result<Vec<usize>, String> {
            Err("unused".into())
        }
    }
    let e = ferrum_gpu_gui::engine::Backend::gpt2_engine_with(
        &NoExtras,
        &model(),
        false,
        &ferrum_gpu_gui::engine::EngineOptions {
            tie: true,
            dropout: 0.0,
            seed: 1,
        },
    )
    .err()
    .unwrap();
    assert!(e.contains("not available on pretend GPU"), "{e}");
}
