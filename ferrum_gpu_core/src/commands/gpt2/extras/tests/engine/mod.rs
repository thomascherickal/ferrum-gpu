use crate::commands::gpt2::extras::{engine, Extras};
use crate::session::{DeviceChoice, Session};
use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::optim::Adam;
use ferrum_ref::trainer::StepConfig;

fn cpu() -> Session {
    Session::new(DeviceChoice::Cpu, None, 1).unwrap()
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

mod without_extras_it_is_the_backends_ordinary_engine;
