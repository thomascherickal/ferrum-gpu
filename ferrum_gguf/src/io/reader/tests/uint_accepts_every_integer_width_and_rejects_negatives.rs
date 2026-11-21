use super::*;

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
