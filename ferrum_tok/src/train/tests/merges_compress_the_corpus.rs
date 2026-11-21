use super::*;

#[test]
fn merges_compress_the_corpus() {
    let tok = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
    assert!(!tok.encode_state().is_empty(), "no merges learned");
    let ids = tok.encode(CORPUS);
    let (n_ids, n_bytes) = (ids.len(), CORPUS.len());
    assert!(n_ids < n_bytes, "{n_ids} ids for {n_bytes} bytes");
}

#[test]
fn vocab_size_is_respected() {
    let tok = ByteBpeTokenizer::train(CORPUS, 260).unwrap();
    assert!(tok.vocab_size() <= 260);
    assert_eq!(tok.vocab_size(), 260); // corpus has ≥4 repeating pairs
}

#[test]
fn training_stops_when_nothing_repeats() {
    // All-distinct bytes: no pair occurs twice, so no merges happen.
    let tok = ByteBpeTokenizer::train("abcdefg", 1000).unwrap();
    assert_eq!(tok.vocab_size(), 257); // 256 bytes + reserved EOS
    assert!(tok.merges.is_empty());
}

#[test]
fn vocab_below_256_errors() {
    assert!(ByteBpeTokenizer::train(CORPUS, 100).is_err());
}

#[test]
fn incremental_training_matches_naive_oracle() {
    // Repeats, overlapping runs ("aaaa"), ties, and multibyte text.
    let corpus = format!(
        "{CORPUS} aaaa aaa abab ababab mississippi banana bandana 🌸🌸 мир мира \n\t {}",
        "the cat sat on the mat with the hat ".repeat(7)
    );
    for vocab in [257, 270, 300, 400] {
        let fast = ByteBpeTokenizer::train(&corpus, vocab).unwrap();
        assert_eq!(fast.merges, naive_merges(&corpus, vocab), "vocab {vocab}");
    }
}

#[test]
fn a_corpus_without_adjacent_pairs_learns_no_merges() {
    for corpus in ["a", "a b"] {
        assert!(naive_merges(corpus, 300).is_empty());
        assert!(ByteBpeTokenizer::train(corpus, 300)
            .unwrap()
            .merges
            .is_empty());
    }
}

#[test]
fn training_is_deterministic() {
    let a = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
    let b = ByteBpeTokenizer::train(CORPUS, 300).unwrap();
    assert_eq!(a.encode_state(), b.encode_state());
    assert_eq!(a.encode(CORPUS), b.encode(CORPUS));
}

#[test]
fn multibyte_utf8_merges_safely() {
    // Repeated multi-byte chars: merges may span partial code points
    // internally, but decode must still reassemble valid UTF-8.
    let corpus = "🌸🌸🌸🌸🌸 zen 🌸🌸🌸";
    let tok = ByteBpeTokenizer::train(corpus, 280).unwrap();
    assert_eq!(tok.decode(&tok.encode(corpus)), corpus);
}
