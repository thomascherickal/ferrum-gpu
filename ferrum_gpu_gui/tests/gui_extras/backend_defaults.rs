//! A backend that implements only the required methods refuses everything else by name: the
//! trait's defaults are what a new backend starts from.

use ferrum_gpu_gui::engine::{Backend, EngineOptions, GenOpts, Progress, TrainSpec};
use ferrum_ref::gpt2::{Config, Gpt2};
use ferrum_ref::llama::Llama;
use std::sync::{Arc, Mutex};

struct Bare;

impl Backend for Bare {
    fn kind(&self) -> &'static str {
        "bare"
    }
    fn name(&self) -> String {
        "the bare backend".into()
    }
    fn train(&self, _: &Llama, _: &TrainSpec, _: &Arc<Mutex<Progress>>) -> Result<Llama, String> {
        Err("unused".into())
    }
    fn generate(&self, _: &Llama, _: &[usize], _: &GenOpts) -> Result<Vec<usize>, String> {
        Err("unused".into())
    }
}

fn tiny_gpt2() -> Gpt2 {
    Gpt2::init(
        Config {
            vocab: 11,
            ctx: 8,
            dim: 8,
            heads: 2,
            layers: 1,
            hidden: 16,
        },
        1,
    )
}

#[test]
fn gpt2_training_is_refused_by_name() {
    let e = Bare.gpt2_engine(&tiny_gpt2(), false).err().unwrap();
    assert!(
        e.contains("GPT-2 training is not available on the bare backend"),
        "{e}"
    );
}

#[test]
fn tied_and_dropout_engines_are_refused_by_name() {
    let o = EngineOptions {
        tie: true,
        dropout: 0.1,
        seed: 1,
    };
    let e = Bare
        .gpt2_engine_with(&tiny_gpt2(), false, &o)
        .err()
        .unwrap();
    assert!(e.contains("not available on the bare backend"), "{e}");
}

#[test]
fn decoding_is_refused_by_name() {
    let e = Bare.gpt2_decoder(&tiny_gpt2()).err().unwrap();
    assert!(
        e.contains("GPT-2 decoding is not available on the bare backend"),
        "{e}"
    );
    let m = ferrum_gpu_gui::engine::fresh_model("tiny", 1).unwrap();
    let e = Bare.llama_decoder(&m).err().unwrap();
    assert!(
        e.contains("decoding is not available on the bare backend"),
        "{e}"
    );
}
