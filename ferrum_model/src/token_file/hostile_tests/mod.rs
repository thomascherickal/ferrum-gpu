//! Token files that are wrong in exactly one way, each refused by name and never read as ids. Every
//! file is built from the same minimal valid one (3 ids), with one field changed.
use super::*;
use ferrum_gguf::{GgufBuilder, Value, GGML_I16, GGML_I32};

/// Which field of the otherwise valid file to break.
#[derive(Clone, Default)]
struct Spec {
    arch: Option<&'static str>,
    version: Option<u32>,
    vocab: Option<u64>,
    skip_sha: bool,
    skip_docs: bool,
    skip_eos_value: bool,
    tensors: Option<usize>, // number of tensors (default 1)
    tensor_type: Option<u32>,
    tensor_dim: Option<u64>,
    tensor_name: Option<&'static str>,
}

fn build(s: &Spec) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.meta(
        "general.architecture",
        Value::Str(s.arch.unwrap_or(TOKEN_FILE_ARCH).into()),
    );
    b.meta(
        "ferrum.tokens.version",
        Value::U32(s.version.unwrap_or(VERSION)),
    );
    b.meta(
        "ferrum.tokens.vocab_size",
        Value::U64(s.vocab.unwrap_or(10)),
    );
    b.meta("ferrum.tokens.token_count", Value::U64(3));
    if !s.skip_docs {
        b.meta("ferrum.tokens.document_count", Value::U64(1));
    }
    if !s.skip_eos_value {
        b.meta("ferrum.tokens.eos_id", Value::U32(2));
    }
    if !s.skip_sha {
        b.meta(
            "ferrum.tokens.tokenizer_sha256",
            Value::Str("ab".repeat(32)),
        );
    }
    let ty = s.tensor_type.unwrap_or(GGML_I16);
    let width = if ty == GGML_I16 { 2 } else { 4 };
    for i in 0..s.tensors.unwrap_or(1) {
        let name = if i == 0 {
            s.tensor_name.unwrap_or(TENSOR)
        } else {
            "extra"
        };
        b.tensor_raw(name, &[s.tensor_dim.unwrap_or(3)], ty, vec![0u8; 3 * width]);
    }
    b.into_bytes()
}

fn open(s: &Spec) -> Result<TokenFile, String> {
    let d = std::env::temp_dir().join(format!("ferrum_tokhostile_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(format!(
        "{:x}.gguf",
        build(s).iter().map(|&b| u64::from(b)).sum::<u64>()
    ));
    std::fs::write(&p, build(s)).unwrap();
    TokenFile::open(p.to_str().unwrap())
}

fn err(s: &Spec) -> String {
    open(s).unwrap_err()
}

mod a_missing_count_is_named;
mod a_tensor_of_the_wrong_type_or_length_is_refused;
mod the_unbroken_file_opens;
