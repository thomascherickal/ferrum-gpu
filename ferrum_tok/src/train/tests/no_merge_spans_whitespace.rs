use super::*;

#[test]
fn no_merge_spans_whitespace() {
    // "ab ab ab" — without pre-tokenization BPE could learn a token that
    // straddles the space (e.g. "b a"); pre-tokenization forbids it.
    let tok = ByteBpeTokenizer::train("ab ab ab ab ab", 300).unwrap();
    for (id, bytes) in tok.vocab.iter().enumerate() {
        if id >= BASE_VOCAB {
            // GPT-2 chunks may carry one leading space, never an inner one.
            let has_inner_space = bytes.iter().skip(1).any(|&b| (b as char).is_whitespace());
            assert!(
                !has_inner_space,
                "merged token {id} ({bytes:?}) spans whitespace"
            );
        }
    }
}

#[test]
fn doc_separator_becomes_eos_when_registered() {
    let corpus = docs_corpus();
    let tok = ByteBpeTokenizer::train_with_docs(&corpus, 300).unwrap();
    assert_eq!(tok.vocab_size(), 300);
    let eos = tok.eos_id().expect("EOS registered");
    assert_eq!(DOC_SEP, TOK_EOS);
    let ids = tok.encode_docs("a.<|endoftext|>b.");
    assert_eq!(ids.iter().filter(|&&i| i == eos).count(), 1);
    assert_eq!(tok.decode(&ids), "a.b.");
    // The separator never leaks into the learned merges.
    assert!(!tok.encode(&corpus).contains(&eos));
    assert!(tok.encode_docs(&corpus).contains(&eos));
}

#[test]
fn train_with_docs_without_separator_equals_plain_training() {
    let corpus = "abc abc abc abd abd abd ".repeat(10);
    let plain = ByteBpeTokenizer::train(&corpus, 262).unwrap();
    let docs = ByteBpeTokenizer::train_with_docs(&corpus, 262).unwrap();
    assert_eq!(plain.encode_state(), docs.encode_state());
    // Vocab below the irreducible minimum still errors through the wrapper.
    assert!(ByteBpeTokenizer::train_with_docs("<|endoftext|>", 256).is_err());
}

#[test]
fn merges_never_cross_pretokenized_chunks() {
    // "ab" appears only across the chunk boundary of "a" + " b"-style splits
    let tok = ByteBpeTokenizer::train(&"a,b ".repeat(30), 300).unwrap();
    let ids = tok.encode("a,b");
    for id in ids {
        let bytes = tok.token_bytes(id).expect("every id has bytes");
        let has_both = bytes.len() > 1 && bytes.contains(&b'a') && bytes.contains(&b',');
        assert!(
            !has_both,
            "token {id} spans letter and punctuation: {bytes:?}"
        );
    }
}
