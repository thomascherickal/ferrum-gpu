use crate::test_support::CORPUS;
use crate::ByteBpeTokenizer;

#[test]
fn trained_roundtrip_exact() {
    let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
    for text in [CORPUS, "lowest news", "unseen zzz 🌸 text"] {
        assert_eq!(
            tok.decode(&tok.encode(text)),
            text,
            "roundtrip failed for {text:?}"
        );
    }
}

#[test]
fn rank_based_encode_is_greedy_lowest_rank_first() {
    // Train so "lo","ow" (hence "low") are learned; encoding "low" must use
    // the merged token, i.e. fewer tokens than its 3 bytes.
    let tok = ByteBpeTokenizer::train("low low low low low", 300).unwrap();
    let ids = tok.encode("low");
    assert!(ids.len() < 3, "expected a merged 'low' token, got {ids:?}");
    assert_eq!(tok.decode(&ids), "low");
}

#[test]
fn encode_uses_learned_merges_across_words_independently() {
    let tok = ByteBpeTokenizer::train("hello hello world world hello world", 320).unwrap();
    // Two words → each encodes via its own within-word merges; concatenation
    // round-trips and the space is its own (unmerged) token between them.
    let ids = tok.encode("hello world");
    assert_eq!(tok.decode(&ids), "hello world");
    assert!(ids.len() <= "hello world".len());
}

#[test]
fn encode_framed_adds_bos_and_eos() {
    let mut tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
    tok.add_standard_special_tokens();
    let bare = tok.encode("lower");
    let framed = tok.encode_framed("lower", true, true);
    assert_eq!(framed.first(), Some(&tok.bos_id().unwrap()));
    assert_eq!(framed.last(), Some(&tok.eos_id().unwrap()));
    assert_eq!(&framed[1..framed.len() - 1], bare.as_slice());
    // Each side is optional.
    assert_eq!(tok.encode_framed("lower", false, false), bare);
    assert_eq!(
        tok.encode_framed("lower", true, false)[0],
        tok.bos_id().unwrap()
    );
    assert_eq!(
        tok.encode_framed("lower", true, false).len(),
        bare.len() + 1
    );
    assert_eq!(
        tok.encode_framed("lower", false, true).len(),
        bare.len() + 1
    );
    // Without registered specials, framing is a no-op.
    let plain = ByteBpeTokenizer::byte_level();
    assert_eq!(
        plain.encode_framed("lower", true, true),
        plain.encode("lower")
    );
}

#[test]
fn encode_docs_separator_positions_and_missing_eos() {
    let tok = ByteBpeTokenizer::train(&"ab ab ab ".repeat(10), 270).unwrap();
    let eos = tok.eos_id().unwrap();
    // Leading, trailing, adjacent and absent separators.
    assert_eq!(tok.encode_docs("<|endoftext|>"), vec![eos]);
    assert_eq!(
        tok.encode_docs("<|endoftext|><|endoftext|>"),
        vec![eos, eos]
    );
    let mixed = tok.encode_docs("<|endoftext|>ab<|endoftext|>");
    assert_eq!(mixed.first(), Some(&eos));
    assert_eq!(mixed.last(), Some(&eos));
    assert_eq!(tok.encode_docs("ab ab"), tok.encode("ab ab"));
    assert!(tok.encode_docs("").is_empty());
    // A tokenizer without EOS keeps the separator as literal text.
    let bare = ByteBpeTokenizer::byte_level();
    assert!(bare.eos_id().is_none());
    assert_eq!(
        bare.encode_docs("x<|endoftext|>y"),
        bare.encode("x<|endoftext|>y")
    );
    assert_eq!(
        bare.decode(&bare.encode_docs("x<|endoftext|>y")),
        "x<|endoftext|>y"
    );
}
