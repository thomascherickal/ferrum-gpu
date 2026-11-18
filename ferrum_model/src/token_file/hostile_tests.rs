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
    b.meta("ferrum.tokens.document_count", Value::U64(1));
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

#[test]
fn the_unbroken_file_opens() {
    let f = open(&Spec::default()).unwrap();
    assert_eq!((f.vocab_size(), f.tokenizer_sha256().len()), (10, 64));
}

#[test]
fn another_architecture_is_not_a_token_file() {
    let e = err(&Spec {
        arch: Some("gpt2"),
        ..Spec::default()
    });
    assert!(e.contains("is not a ferrum-tokens file"), "{e}");
}

#[test]
fn an_unknown_version_is_refused() {
    let e = err(&Spec {
        version: Some(99),
        ..Spec::default()
    });
    assert!(e.contains("unsupported token file version"), "{e}");
}

#[test]
fn a_zero_or_oversized_vocabulary_is_refused() {
    for v in [0u64, i32::MAX as u64 + 1] {
        let e = err(&Spec {
            vocab: Some(v),
            ..Spec::default()
        });
        assert!(
            e.contains("vocab_size") && e.contains("out of range"),
            "{v}: {e}"
        );
    }
}

#[test]
fn a_missing_tokenizer_hash_is_refused() {
    let e = err(&Spec {
        skip_sha: true,
        ..Spec::default()
    });
    assert!(
        e.contains("tokenizer_sha256") && e.contains("missing"),
        "{e}"
    );
}

#[test]
fn an_end_of_text_id_outside_the_vocabulary_is_refused() {
    // eos_id 2 with a vocabulary of 2 ids
    let e = err(&Spec {
        vocab: Some(2),
        ..Spec::default()
    });
    assert!(e.contains("eos_id is outside the vocabulary"), "{e}");
}

#[test]
fn zero_or_several_tensors_are_refused() {
    let e = err(&Spec {
        tensors: Some(0),
        ..Spec::default()
    });
    assert!(e.contains("expected exactly one 'tokens' tensor"), "{e}");
    let e = err(&Spec {
        tensors: Some(2),
        ..Spec::default()
    });
    assert!(e.contains("expected exactly one 'tokens' tensor"), "{e}");
    let e = err(&Spec {
        tensor_name: Some("ids"),
        ..Spec::default()
    });
    assert!(e.contains("expected exactly one 'tokens' tensor"), "{e}");
}

#[test]
fn a_tensor_of_the_wrong_type_or_length_is_refused() {
    // a vocabulary of 10 needs 16-bit ids; 32-bit ids belong to vocabularies over 65535
    let e = err(&Spec {
        tensor_type: Some(GGML_I32),
        ..Spec::default()
    });
    assert!(e.contains("must be 1-D [3] of type"), "{e}");
    let e = err(&Spec {
        tensor_dim: Some(4),
        ..Spec::default()
    });
    assert!(e.contains("must be 1-D [3] of type"), "{e}");
}

#[test]
fn a_file_that_is_not_gguf_is_not_a_token_file() {
    let d = std::env::temp_dir().join(format!("ferrum_tokhostile_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join("text.txt");
    std::fs::write(&p, "just some text, not a token file at all").unwrap();
    assert!(!is_token_file(p.to_str().unwrap()));
    assert!(TokenFile::open(p.to_str().unwrap()).is_err());
    assert!(!is_token_file("/nonexistent/ferrum/file"));
    // too short to hold a header
    std::fs::write(&p, b"GGUF").unwrap();
    assert!(!is_token_file(p.to_str().unwrap()));
    // the right magic with a version GGUF does not have
    std::fs::write(&p, b"GGUF\x09\0\0\0rest of the file").unwrap();
    assert!(!is_token_file(p.to_str().unwrap()));
}

#[test]
fn a_skipped_eos_value_is_just_no_eos() {
    let f = open(&Spec {
        skip_eos_value: true,
        ..Spec::default()
    })
    .unwrap();
    assert_eq!(f.eos_id(), None);
}
