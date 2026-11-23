//! Starting a server from flags, for both kinds of model file and the ways it can be refused.
use super::*;
use ferrum_model::gpt2_gguf::{to_gguf_with, Quant};
use ferrum_model::llama_gguf::to_gguf;
use ferrum_ref::gpt2::{Config as GptConfig, Gpt2};
use ferrum_ref::llama::{Config, Llama, RopeType};
use std::path::PathBuf;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_boot_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn argv(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn gpt2_file(d: &std::path::Path) -> String {
    let cfg = GptConfig {
        vocab: 256,
        ctx: 16,
        dim: 16,
        heads: 2,
        layers: 1,
        hidden: 32,
    };
    let p = d.join("g.gguf");
    std::fs::write(&p, to_gguf_with(&Gpt2::init(cfg, 1), Quant::F32, None)).unwrap();
    p.display().to_string()
}

fn llama_file(d: &std::path::Path) -> String {
    let cfg = Config {
        vocab: 256,
        ctx: 16,
        dim: 16,
        heads: 2,
        kv_heads: 1,
        head_dim: 8,
        layers: 1,
        hidden: 32,
        rope_dim: 8,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    };
    let p = d.join("l.gguf");
    std::fs::write(
        &p,
        to_gguf(&Llama::random(cfg, 1, false, false), Quant::F32),
    )
    .unwrap();
    p.display().to_string()
}

mod both_kinds_of_model_start_a_server_and_say_what_they_started;
mod the_token_comes_from_the_flag_then_the_environment;
