use super::*;
use crate::reader::Gguf;
use crate::{GGML_Q4_0, GGML_Q8_0};

#[test]
fn empty_file_parses() {
    let g = Gguf::parse(GgufBuilder::new().into_bytes()).unwrap();
    assert!(g.kv.is_empty() && g.tensors.is_empty());
}

#[test]
fn every_value_type_round_trips() {
    let vals = vec![
        Value::U8(200),
        Value::I8(-5),
        Value::U16(60000),
        Value::I16(-30000),
        Value::U32(4_000_000_000),
        Value::I32(-7),
        Value::F32(1.5),
        Value::Bool(true),
        Value::Str("héllo".into()),
        Value::Arr(vec![Value::I32(1), Value::I32(2)]),
        Value::U64(u64::MAX),
        Value::I64(i64::MIN),
        Value::F64(2.25),
    ];
    let mut b = GgufBuilder::new();
    for (i, v) in vals.iter().enumerate() {
        b.meta(&format!("k{i}"), v.clone());
    }
    let g = Gguf::parse(b.into_bytes()).unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(g.get(&format!("k{i}")), Some(v), "k{i}");
    }
}

#[test]
fn tensors_are_aligned_and_independent() {
    let mut b = GgufBuilder::new();
    b.tensor_f32("a", &[3], &[1.0, 2.0, 3.0]); // 12 bytes, padded to 32
    b.tensor_f32("b", &[1], &[9.0]);
    let g = Gguf::parse(b.into_bytes()).unwrap();
    assert_eq!(g.tensor("a").unwrap().offset, 0);
    assert_eq!(g.tensor("b").unwrap().offset, 32);
    assert_eq!(g.tensor_f32("b").unwrap(), vec![9.0]);
}

#[test]
fn quantised_tensors_load_back_close() {
    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) / 9.0).collect();
    let mut b = GgufBuilder::new();
    b.tensor_raw("q8", &[32, 2], GGML_Q8_0, encode(GGML_Q8_0, &data).unwrap());
    b.tensor_raw("q4", &[32, 2], GGML_Q4_0, encode(GGML_Q4_0, &data).unwrap());
    let g = Gguf::parse(b.into_bytes()).unwrap();
    let (q8, q4) = (g.tensor_f32("q8").unwrap(), g.tensor_f32("q4").unwrap());
    let err = |q: &[f32]| {
        q.iter()
            .zip(&data)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
    };
    assert!(err(&q8) < 0.03, "q8 err {}", err(&q8));
    assert!(err(&q4) < 0.5, "q4 err {}", err(&q4));
    assert!(err(&q8) < err(&q4));
}
