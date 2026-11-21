use super::*;
use crate::engine::{fresh_model, CpuBackend};
use ferrum_model::gpt2_gguf::{from_gguf, to_gguf_with, Quant};
use ferrum_ref::gpt2::Config;

fn req(max_new: usize) -> GenReq {
    GenReq {
        max_new,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 1,
        stops: vec![],
    }
}

fn gpt2() -> Gpt2Decoder {
    let tok = ByteBpeTokenizer::train_with_docs(&"abab cdcd low lower ".repeat(10), 280).unwrap();
    let cfg = Config {
        vocab: tok.vocab_size(),
        ctx: 16,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 16,
    };
    let bytes = to_gguf_with(&Gpt2::init(cfg, 2), Quant::F32, Some(&tok));
    let g = ferrum_gguf::Gguf::parse(bytes).unwrap();
    let model = from_gguf(&g).unwrap();
    Gpt2Decoder::new(Arc::new(CpuBackend::new(1)), Loaded { model, tok })
}

mod cut_at_stop_takes_the_earliest_stop;
mod stub_is_deterministic;
