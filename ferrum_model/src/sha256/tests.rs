use super::*;

#[test]
fn a_default_hasher_is_a_new_one() {
    assert_eq!(Sha256::default().hex(), sha256_hex(b""));
}

#[test]
fn incremental_updates_equal_one_shot_at_every_split() {
    let data: Vec<u8> = (0..300u32).map(|i| (i * 7) as u8).collect();
    for cut in [0, 1, 55, 56, 63, 64, 65, 127, 128, 299, 300] {
        let mut s = Sha256::new();
        s.update(&data[..cut]);
        s.update(&data[cut..]);
        assert_eq!(s.hex(), sha256_hex(&data), "cut {cut}");
    }
}

#[test]
fn matches_the_published_test_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    assert_eq!(
        sha256_hex(&vec![b'a'; 1_000_000]),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn padding_boundaries_and_the_fixture_tokenizer_file() {
    // lengths around the 55/56/64 byte padding edges all hash to 64 hex digits and differ
    let mut seen = std::collections::HashSet::new();
    for n in [54, 55, 56, 57, 63, 64, 65, 119, 120] {
        let d = sha256_hex(&vec![7u8; n]);
        assert_eq!(d.len(), 64);
        assert!(seen.insert(d), "length {n}");
    }
    // ferrum-cpu recorded this digest for its own tokenizer file in the token file it wrote
    let tok = include_bytes!("../../../fixtures/bpe/tok.gguf");
    assert_eq!(
        sha256_hex(tok),
        "ca035b85db47168a4db57cff2d818fd0078954949f17b30e126c38892be653b7"
    );
}
