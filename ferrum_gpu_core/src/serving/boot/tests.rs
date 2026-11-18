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

#[test]
fn both_kinds_of_model_start_a_server_and_say_what_they_started() {
    let d = tmp("both");
    for model in [gpt2_file(&d), llama_file(&d)] {
        let mut out = Vec::new();
        let b = boot(
            &argv(&format!("--model {model} --port 0 --device auto --token t")),
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("device:") && text.contains("ferrum-gpu server: http://"),
            "{text}"
        );
        assert!(text.contains("bearer token required"), "{text}");
        assert!(b.has_token && b.addr.port() != 0);
        b.server.stop();
    }
}

#[test]
fn a_server_without_a_token_says_nothing_about_auth() {
    let d = tmp("notoken");
    let mut out = Vec::new();
    let b = boot(
        &argv(&format!("--model {} --port 0 --device cpu", gpt2_file(&d))),
        &mut out,
    )
    .unwrap();
    assert!(!String::from_utf8(out).unwrap().contains("bearer"));
    assert!(!b.has_token);
    b.server.stop();
}

#[test]
fn what_cannot_start_is_refused_with_a_reason() {
    let d = tmp("refused");
    let (g, l) = (gpt2_file(&d), llama_file(&d));
    for (args, why) in [
        (
            format!("--model {g} --device gpu --gpu-adapter no-such-adapter-xyz"),
            "adapter",
        ),
        (format!("--model {g} --port notaport"), "--port"),
        (format!("--model {g} --host 0.0.0.0 --port 0"), "loopback"),
        ("--port 0".to_string(), "--model"),
        (
            format!("--model {}/missing.gguf --port 0", d.display()),
            "missing.gguf",
        ),
        (
            format!("--model {g} --device gpu --gpu-mem-mb 0 --port 0"),
            "",
        ),
        (
            format!("--model {l} --device gpu --gpu-mem-mb 0 --port 0"),
            "",
        ),
    ] {
        let e = boot(&argv(&args), &mut Vec::new())
            .err()
            .unwrap_or_else(|| panic!("{args} started"));
        assert!(e.contains(why), "{args}: {e}");
    }
}
