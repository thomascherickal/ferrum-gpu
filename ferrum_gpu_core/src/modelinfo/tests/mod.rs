use super::scan::{scan, scan_with, walk, MAX_DEPTH};
use super::*;
use ferrum_gguf::GgufBuilder;
use ferrum_test_support::link;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

static N: AtomicU32 = AtomicU32::new(0);

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ferrum_modelinfo_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A small GGUF: a string, a number, a long string, an array, an f32 tensor and a Q8_0 tensor.
fn model_bytes() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    b.meta("general.name", Value::Str("tiny test".into()));
    b.meta("gpt2.context_length", Value::U32(64));
    b.meta("note", Value::Str("x".repeat(500)));
    b.meta("tokens", Value::Arr((0..20).map(Value::U32).collect()));
    b.tensor_f32("token_embd.weight", &[4, 3], &[0.5; 12]);
    b.tensor_raw("blk.0.ffn.weight", &[32, 2], 8, vec![0u8; 2 * 34]);
    b.into_bytes()
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, bytes).unwrap();
    p
}

mod a_file_that_cannot_be_opened_is_named_with_the_reason;
mod a_model_is_described_by_its_header_metadata_and_tensors;
mod a_model_with_no_name_or_architecture_still_inspects;
mod a_scan_stops_at_the_file_limit_and_at_the_depth_limit;
mod files_that_are_not_models_are_refused_with_the_path_in_the_message;
