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
