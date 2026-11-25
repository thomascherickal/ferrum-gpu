use super::*;
use crate::test_support::{field, reply, request};
use ferrum_gguf::{GgufBuilder, Value};
use std::sync::atomic::{AtomicU32, Ordering};

static N: AtomicU32 = AtomicU32::new(0);

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ferrum_modelsapi_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn model(dir: &std::path::Path, rel: &str) -> PathBuf {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    b.tensor_f32("w", &[4], &[1.0; 4]);
    let p = dir.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, b.into_bytes()).unwrap();
    p
}

fn get(path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    call("GET", path, query)
}

fn call(method: &str, path: &str, query: &[(&str, &str)]) -> (u16, Json) {
    reply(route(&request(method, path, query, &[])))
}

mod only_the_gguf_paths_are_handled;
mod the_api_router_sends_gguf_paths_here;
