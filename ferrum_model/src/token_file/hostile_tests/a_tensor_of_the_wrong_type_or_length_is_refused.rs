use super::*;

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
