use super::*;

#[test]
fn a_model_is_described_by_its_header_metadata_and_tensors() {
    let d = tmp();
    let p = write(&d, "m.gguf", &model_bytes());
    let i = inspect(&p).unwrap();
    assert_eq!(i.version, 3);
    assert_eq!(i.architecture.as_deref(), Some("gpt2"));
    assert_eq!(i.name.as_deref(), Some("tiny test"));
    assert_eq!((i.n_tensors, i.n_kv), (2, 5));
    assert_eq!(i.parameters, 12 + 64);
    assert_eq!(i.tensor_bytes, 12 * 4 + 2 * 34);
    assert_eq!(i.file_size, std::fs::metadata(&p).unwrap().len());
    assert!(!i.truncated);
    assert_eq!(i.tensors[0].name, "token_embd.weight");
    assert_eq!(i.tensors[0].dims, [4, 3]);
    assert_eq!(
        (i.tensors[0].kind.as_str(), i.tensors[0].bytes),
        ("F32", Some(48))
    );
    assert_eq!(
        (i.tensors[1].kind.as_str(), i.tensors[1].bytes),
        ("Q8_0", Some(68))
    );
    // the kinds are totalled, biggest first
    assert_eq!(
        i.kinds,
        vec![
            KindTotal {
                kind: "Q8_0".into(),
                tensors: 1,
                bytes: 68
            },
            KindTotal {
                kind: "F32".into(),
                tensors: 1,
                bytes: 48
            },
        ]
    );
}

#[test]
fn metadata_values_are_shown_cut_and_summarised() {
    let d = tmp();
    let i = inspect(&write(&d, "m.gguf", &model_bytes())).unwrap();
    let get = |k: &str| i.metadata.iter().find(|(n, _)| n == k).unwrap().1.clone();
    assert_eq!(get("general.architecture"), "\"gpt2\"");
    assert_eq!(get("gpt2.context_length"), "64");
    let note = get("note");
    assert!(
        note.ends_with('…') && note.chars().count() <= MAX_VALUE_CHARS + 1,
        "{note}"
    );
    let toks = get("tokens");
    assert!(toks.starts_with("[20 items] 0, 1, 2, 3, 4, 5"), "{toks}");
    // every value type has a display
    for (v, want) in [
        (Value::U8(1), "1"),
        (Value::I8(-2), "-2"),
        (Value::U16(3), "3"),
        (Value::I16(-4), "-4"),
        (Value::U32(5), "5"),
        (Value::I32(-6), "-6"),
        (Value::U64(7), "7"),
        (Value::I64(-8), "-8"),
        (Value::F32(1.5), "1.5"),
        (Value::F64(2.5), "2.5"),
        (Value::Bool(true), "true"),
    ] {
        assert_eq!(show(&v), want);
    }
}

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
