use super::*;

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
