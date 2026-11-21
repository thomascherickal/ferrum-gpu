use super::*;

#[test]
fn bpe_applies_merges_in_rank_order_and_decodes_exactly() {
    let t = bpe();
    assert_eq!(t.model(), TokModel::Bpe);
    assert_eq!(t.encode("hello"), vec![8]);
    assert_eq!(t.encode("hell"), vec![7]);
    assert_eq!(t.encode(" hello"), vec![4, 8]); // the space is its own piece: no Ġh merge
    assert_eq!(t.decode(&[8, 4, 8]), "hello hello");
    // a character with no token is dropped rather than panicking
    assert_eq!(t.encode("z"), Vec::<usize>::new());
    // special-token text decodes to itself, ids out of range are skipped
    assert_eq!(t.decode(&[9, 999]), "<|im_end|>");
    assert_eq!(
        (t.vocab_size(), t.token_id("he"), t.token_id("zz")),
        (10, Some(5), None)
    );
    assert_eq!((t.bos(), t.eos()), (Some(9), Some(9)));
}

#[test]
fn a_merged_piece_the_vocabulary_lacks_is_spelled_by_its_characters() {
    // "h e" merges to "he", which is not a token: the pieces fall back to "h" and "e"
    let t = build("gpt2", &["h", "e"], &["h e"], &[]).unwrap();
    assert_eq!(t.encode("he"), vec![0, 1]);
}

#[test]
fn pretokenizer_groups_letters_digits_symbols_and_whitespace() {
    let chunks = |s: &str| pretokenize(s);
    assert_eq!(chunks("ab cd"), vec!["ab", " cd"]);
    assert_eq!(chunks("a1  b"), vec!["a", "1", "  ", "b"]);
    assert_eq!(chunks("x,.y"), vec!["x", ",.", "y"]);
    assert_eq!(chunks("a "), vec!["a", " "]);
    assert_eq!(chunks("a\n\nb"), vec!["a", "\n\n", "b"]);
    assert!(chunks("").is_empty());
}

#[test]
fn spm_uses_longest_match_and_byte_fallback() {
    let t = build(
        "llama",
        &["<unk>", "\u{2581}", "\u{2581}a", "b", "<0x78>", "<s>"],
        &[],
        &[("tokenizer.ggml.bos_token_id", Value::U32(5))],
    )
    .unwrap();
    assert_eq!(t.model(), TokModel::Spm);
    assert_eq!(t.encode("ab"), vec![2, 3]);
    assert_eq!(t.encode("x"), vec![1, 4]); // "▁", then the byte-fallback token for 'x'
    assert_eq!(t.decode(&[2, 3]), "ab"); // the dummy-prefix space is dropped
    assert_eq!(t.decode(&[1, 4]), "x");
    assert_eq!(t.decode(&[2, 2]), "a a"); // spaces inside survive
                                          // SPM adds BOS by default
    assert_eq!(t.prefix_bos(), Some(5));
}
