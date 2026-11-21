use super::*;

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
