use super::*;
use crate::writer::GgufBuilder;
use crate::GGML_F32;

fn sample() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    b.meta("gpt2.block_count", Value::U32(2));
    b.meta("x.eps", Value::F32(1e-5));
    b.meta(
        "x.list",
        Value::Arr(vec![Value::Str("a".into()), Value::Str("bc".into())]),
    );
    b.tensor_f32("w", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    b.tensor_f32("b", &[2], &[0.5, -0.5]);
    b.into_bytes()
}

#[test]
fn a_header_only_parse_checks_extents_against_the_file_length() {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("x".into()));
    b.tensor_f32("w", &[100], &[1.0; 100]);
    let bytes = b.into_bytes();
    let total = bytes.len() as u64;
    let head_len = bytes.len() - 400;
    let head = bytes[..head_len].to_vec();
    let g = Gguf::parse_prefix(head.clone(), total).unwrap();
    assert_eq!(g.tensor("w").unwrap().dims, vec![100]);
    assert!(g.data_offset() <= head_len && g.data_offset().is_multiple_of(32));
    assert!(g.tensor_bytes("w").unwrap_err().0.contains("header-only"));
    // the same header against a shorter file is a truncated file
    assert!(Gguf::parse_prefix(head, g.data_offset() as u64 + 399)
        .unwrap_err()
        .0
        .contains("extends past"));
}

#[test]
fn read_only_quantised_tensors_load_as_f32() {
    use crate::f16::f32_to_f16;
    use crate::{GGML_Q4_1, GGML_Q6_K};
    // one Q4_1 block (d = 0.5, m = 1, nibbles 3) and one Q6_K block (constant 8.0)
    let mut q41 = [f32_to_f16(0.5).to_le_bytes(), f32_to_f16(1.0).to_le_bytes()].concat();
    q41.extend([0x33u8; 16]);
    let mut q6 = vec![0x88u8; 128];
    q6.extend([0xAAu8; 64]);
    q6.extend(std::iter::repeat_n(2u8, 16));
    q6.extend(f32_to_f16(0.5).to_le_bytes());
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("llama".into()));
    b.tensor_raw("a", &[32], GGML_Q4_1, q41);
    b.tensor_raw("b", &[256], GGML_Q6_K, q6);
    let g = Gguf::parse(b.into_bytes()).unwrap();
    assert!(g
        .tensor_f32("a")
        .unwrap()
        .iter()
        .all(|&v| (v - 2.5).abs() < 1e-3));
    let six = g.tensor_f32("b").unwrap();
    assert_eq!(six.len(), 256);
    assert!(six.iter().all(|&v| (v - 8.0).abs() < 1e-3));
}

#[test]
fn raw_bytes_are_the_stored_encoding() {
    let g = Gguf::parse(sample()).unwrap();
    let raw = g.tensor_bytes("b").unwrap();
    assert_eq!(
        raw,
        [0.5f32.to_le_bytes(), (-0.5f32).to_le_bytes()].concat()
    );
    assert_eq!(g.tensor_bytes("w").unwrap().len(), 24);
    assert!(g
        .tensor_bytes("nope")
        .unwrap_err()
        .to_string()
        .contains("missing tensor"));
}

#[test]
fn round_trips_metadata_and_tensors() {
    let g = Gguf::parse(sample()).unwrap();
    assert_eq!(g.string("general.architecture").unwrap(), "gpt2");
    assert_eq!(g.uint("gpt2.block_count").unwrap(), 2);
    assert_eq!(g.float("x.eps").unwrap(), 1e-5);
    assert_eq!(
        g.get("x.list"),
        Some(&Value::Arr(vec![
            Value::Str("a".into()),
            Value::Str("bc".into())
        ]))
    );
    assert_eq!(g.tensor("w").unwrap().dims, vec![3, 2]);
    assert_eq!(g.tensor("w").unwrap().ggml_type, GGML_F32);
    assert_eq!(
        g.tensor_f32("w").unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(g.tensor_f32("b").unwrap(), vec![0.5, -0.5]);
}

#[test]
fn typed_getters_report_what_is_wrong() {
    let g = Gguf::parse(sample()).unwrap();
    assert!(g
        .uint("missing")
        .unwrap_err()
        .to_string()
        .contains("missing key"));
    assert!(g
        .uint("general.architecture")
        .unwrap_err()
        .to_string()
        .contains("non-negative integer"));
    assert!(g.float("gpt2.block_count").is_err());
    assert!(g.string("gpt2.block_count").is_err());
    assert!(g
        .tensor_f32("nope")
        .unwrap_err()
        .to_string()
        .contains("missing tensor"));
    assert!(g.tensor("nope").is_none());
}

#[test]
fn rejects_bad_headers() {
    let good = sample();
    let mut bad_magic = good.clone();
    bad_magic[0] = b'X';
    assert!(Gguf::parse(bad_magic)
        .unwrap_err()
        .to_string()
        .contains("magic"));
    let mut v2 = good.clone();
    v2[4] = 2;
    assert!(Gguf::parse(v2).unwrap_err().to_string().contains("version"));
    assert!(Gguf::parse(vec![]).is_err());
    assert!(Gguf::parse(good[..10].to_vec()).is_err());
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    let good = sample();
    // The last tensor ("b") holds 8 real bytes followed by 24 bytes of alignment padding;
    // cutting only padding leaves a valid file, so stop before it.
    for cut in 0..good.len() - 24 {
        assert!(Gguf::parse(good[..cut].to_vec()).is_err(), "cut at {cut}");
    }
}

#[test]
fn hostile_counts_do_not_allocate_or_panic() {
    let mut huge_kv = sample();
    huge_kv[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Gguf::parse(huge_kv)
        .unwrap_err()
        .to_string()
        .contains("exceed"));
    let mut huge_tensors = sample();
    huge_tensors[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Gguf::parse(huge_tensors).is_err());
}

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
