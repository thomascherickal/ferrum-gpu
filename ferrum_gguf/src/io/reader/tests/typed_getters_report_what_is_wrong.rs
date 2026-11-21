use super::*;

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
