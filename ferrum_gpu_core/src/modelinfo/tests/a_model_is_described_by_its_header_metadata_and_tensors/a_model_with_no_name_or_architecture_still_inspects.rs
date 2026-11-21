use super::*;

#[test]
fn a_model_with_no_name_or_architecture_still_inspects() {
    let d = tmp();
    let mut b = GgufBuilder::new();
    b.tensor_f32("w", &[2], &[1.0, 2.0]);
    let i = inspect(&write(&d, "bare.gguf", &b.into_bytes())).unwrap();
    assert_eq!((i.architecture, i.name, i.n_kv), (None, None, 0));
    // a non-string architecture is not an architecture
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::U32(1));
    assert_eq!(
        inspect(&write(&d, "odd.gguf", &b.into_bytes()))
            .unwrap()
            .architecture,
        None
    );
}

#[test]
fn only_the_start_of_a_big_file_is_read_and_a_header_over_the_limit_is_said_to_be_so() {
    let d = tmp();
    // 5 MB of "tensor data" behind a header of a few hundred bytes: a 4 KiB prefix is enough
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    b.tensor_raw("big", &[32, 150_000], 8, vec![0u8; 150_000 * 34]);
    let p = write(&d, "big.gguf", &b.into_bytes());
    let i = inspect_with(&p, 4096).unwrap();
    assert_eq!(i.parameters, 32 * 150_000);
    assert!(i.file_size > 5_000_000);
    // a header that does not fit in the prefix is reported as exactly that
    let e = inspect_with(&p, 40).unwrap_err();
    assert!(e.contains("big.gguf"), "{e}");
    assert!(e.contains("larger than the 40 bytes"), "{e}");
}

#[test]
fn the_tensor_list_is_capped_but_the_totals_count_everything() {
    let d = tmp();
    let mut b = GgufBuilder::new();
    for i in 0..(MAX_ROWS + 25) {
        b.tensor_f32(&format!("t{i}"), &[1], &[0.0]);
    }
    let info = inspect(&write(&d, "many.gguf", &b.into_bytes())).unwrap();
    assert_eq!(info.n_tensors, MAX_ROWS + 25);
    assert_eq!(info.tensors.len(), MAX_ROWS);
    assert!(info.truncated);
    assert_eq!(info.parameters, (MAX_ROWS + 25) as u64);
    assert_eq!(info.kinds[0].tensors, MAX_ROWS + 25);
}

#[test]
fn a_tensor_type_the_reader_does_not_support_is_refused_not_guessed_at() {
    let d = tmp();
    let mut b = GgufBuilder::new();
    b.tensor_raw("mystery", &[4], 999, vec![0u8; 16]);
    let e = inspect(&write(&d, "odd.gguf", &b.into_bytes())).unwrap_err();
    assert!(
        e.contains("odd.gguf") && e.contains("unsupported ggml tensor type 999"),
        "{e}"
    );
}
