use crate::test_support::CORPUS;
use crate::{ByteBpeTokenizer, TOK_EOS};

#[test]
fn state_roundtrip_preserves_encoding() {
    let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
    let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
    assert_eq!(restored.vocab_size(), tok.vocab_size());
    assert_eq!(restored.encode(CORPUS), tok.encode(CORPUS));
    assert_eq!(restored.decode(&restored.encode("lowest")), "lowest");
}

#[test]
fn bad_state_errors() {
    assert!(ByteBpeTokenizer::from_state("not-a-pair").is_err());
    assert!(ByteBpeTokenizer::from_state("1,2;9999,3").is_err()); // undefined id
    assert!(ByteBpeTokenizer::from_state("x,y").is_err());
    let e = ByteBpeTokenizer::from_state("1,y").err().unwrap();
    assert!(e.to_string().contains("bad merge id \"y\""), "{e}");
}

#[test]
fn doubling_merge_chain_in_state_is_capped() {
    // "97,97;256,256;257,257;…": token 256+i is 2^(i+1) bytes, so 40
    // entries (~400 bytes) would build a 1 TiB vocab. Rejected near the
    // 64 MiB floor of the cap instead, fast.
    let chain = |n: u32| {
        std::iter::once("97,97".to_string())
            .chain((0..n - 1).map(|i| format!("{0},{0}", 256 + i)))
            .collect::<Vec<_>>()
            .join(";")
    };
    let t = std::time::Instant::now();
    let e = ByteBpeTokenizer::from_state(&chain(40)).err().unwrap();
    assert!(e.to_string().contains("tokenizer state vocab bytes"), "{e}");
    assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
    // A 20-long chain (1 MiB token, 2 MiB total) is within the cap.
    let tok = ByteBpeTokenizer::from_state(&chain(20)).unwrap();
    assert_eq!(tok.vocab.last().unwrap().len(), 1 << 20);
}

#[test]
fn encode_matches_after_state_roundtrip_with_ranks() {
    // The rank map must be rebuilt by from_state so encoding is identical.
    let tok = ByteBpeTokenizer::train(CORPUS, 320).unwrap();
    let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
    for text in [CORPUS, "lowest newer news", "🌸 unseen"] {
        assert_eq!(
            restored.encode(text),
            tok.encode(text),
            "mismatch for {text:?}"
        );
    }
}

#[test]
fn special_tokens_survive_state_roundtrip() {
    let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
    tok.add_standard_special_tokens();
    let restored = ByteBpeTokenizer::from_state(&tok.encode_state()).unwrap();
    assert_eq!(restored.vocab_size(), tok.vocab_size());
    assert_eq!(restored.eos_id(), tok.eos_id());
    assert_eq!(restored.num_special(), 4);
    assert_eq!(restored.encode(CORPUS), tok.encode(CORPUS));

    // Specials also round-trip on a byte-level (no-merge) tokenizer.
    let mut bl = ByteBpeTokenizer::byte_level();
    bl.add_special_token(TOK_EOS).unwrap();
    let r = ByteBpeTokenizer::from_state(&bl.encode_state()).unwrap();
    assert_eq!(r.eos_id(), bl.eos_id());
    assert_eq!(r.vocab_size(), 256 + 1);
}
