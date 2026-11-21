use super::*;

#[test]
fn hostile_array_and_tensor_sizes_are_rejected() {
    let mut b = GgufBuilder::new();
    b.meta("a", Value::Arr(vec![Value::U8(1)]));
    let mut bytes = b.into_bytes();
    // The array length field sits after: header(24) + key(8+1) + type(4) + elem type(4).
    let at = 24 + 9 + 4 + 4;
    bytes[at..at + 8].copy_from_slice(&(1u64 << 60).to_le_bytes());
    assert!(Gguf::parse(bytes)
        .unwrap_err()
        .to_string()
        .contains("array length"));

    let mut t = GgufBuilder::new();
    t.tensor_raw("w", &[1 << 40, 1 << 40], GGML_F32, vec![0; 4]);
    assert!(Gguf::parse(t.into_bytes()).is_err());
    let mut t = GgufBuilder::new();
    t.tensor_raw("w", &[1000], GGML_F32, vec![0; 4]);
    assert!(Gguf::parse(t.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("past the end"));
}

#[test]
fn duplicate_names_and_bad_dims_are_rejected() {
    let mut b = GgufBuilder::new();
    b.tensor_f32("w", &[1], &[1.0]);
    b.tensor_f32("w", &[1], &[2.0]);
    assert!(Gguf::parse(b.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
    let mut z = GgufBuilder::new();
    z.tensor_raw("w", &[], GGML_F32, vec![]);
    assert!(Gguf::parse(z.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("dims"));
}

#[test]
fn custom_alignment_is_honoured() {
    let mut b = GgufBuilder::new();
    b.meta("general.alignment", Value::U32(64));
    b.tensor_f32("w", &[2], &[7.0, 8.0]);
    let g = Gguf::parse(b.into_bytes()).unwrap();
    assert_eq!(g.tensor_f32("w").unwrap(), vec![7.0, 8.0]);
    let mut bad = GgufBuilder::new();
    bad.meta("general.alignment", Value::U32(0));
    assert!(Gguf::parse(bad.into_bytes()).is_err());
}

#[test]
fn debug_output_summarises_without_dumping_bytes() {
    let g = Gguf::parse(sample()).unwrap();
    let s = format!("{g:?}");
    assert!(s.contains("4 keys") && s.contains("2 tensors"), "{s}");
    assert!(s.len() < 100);
}

#[test]
fn nested_arrays_and_unknown_value_types_are_rejected() {
    let mut b = GgufBuilder::new();
    b.meta("a", Value::Arr(vec![Value::Arr(vec![Value::U8(1)])]));
    assert!(Gguf::parse(b.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("nested arrays"));
    let mut b = GgufBuilder::new();
    b.meta("a", Value::U8(1));
    let mut bytes = b.into_bytes();
    // The value type tag follows the header (24 bytes) and the key (8-byte length + "a").
    bytes[33..37].copy_from_slice(&99u32.to_le_bytes());
    assert!(Gguf::parse(bytes)
        .unwrap_err()
        .to_string()
        .contains("unknown metadata value type 99"));
}

#[test]
fn uint_accepts_every_integer_width_and_rejects_negatives() {
    let mut b = GgufBuilder::new();
    for (k, v) in [
        ("u8", Value::U8(7)),
        ("u16", Value::U16(300)),
        ("u32", Value::U32(70_000)),
        ("u64", Value::U64(5_000_000_000)),
        ("i32", Value::I32(12)),
        ("i64", Value::I64(34)),
        ("neg32", Value::I32(-1)),
        ("neg64", Value::I64(-1)),
    ] {
        b.meta(k, v);
    }
    let g = Gguf::parse(b.into_bytes()).unwrap();
    for (k, want) in [
        ("u8", 7usize),
        ("u16", 300),
        ("u32", 70_000),
        ("u64", 5_000_000_000),
        ("i32", 12),
        ("i64", 34),
    ] {
        assert_eq!(g.uint(k).unwrap(), want, "{k}");
    }
    assert!(g.uint("neg32").is_err() && g.uint("neg64").is_err());
}

#[test]
fn element_count_overflow_is_an_error() {
    let t = TensorInfo {
        name: "x".into(),
        dims: vec![u64::MAX, 2],
        ggml_type: GGML_F32,
        offset: 0,
    };
    assert!(t.elements().is_err());
}
