use crate::error::InferError;
use crate::test_support::{docs_corpus, naive_merges, small_tok, CORPUS};
use crate::{ByteBpeTokenizer, BASE_VOCAB, DOC_SEP, TOK_EOS};

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

#[test]
fn eos_reservation_edge_cases() {
    // 257 = 256 bytes + EOS and nothing else
    let t = ByteBpeTokenizer::train("aaaa bbbb", 257).unwrap();
    assert_eq!(
        (t.vocab_size(), t.eos_id(), t.num_special()),
        (257, Some(256), 1)
    );
    assert_eq!(t.gguf_merges(), Vec::<String>::new());
    assert_eq!(t.gguf_token_types().last(), Some(&3));
    // EOS lands directly after the last merge, wherever training stopped
    let t = ByteBpeTokenizer::train("ab", 400).unwrap();
    assert_eq!(
        t.vocab_size(),
        257,
        "no repeated pair -> no merges, still EOS"
    );
    assert_eq!(t.eos_id(), Some(256));
    let t = small_tok();
    assert_eq!(t.eos_id(), Some(t.vocab_size() - 1));
    assert!(t.vocab_size() <= 290);
    assert_eq!(
        t.decode(&[t.eos_id().unwrap()]),
        "",
        "EOS decodes to nothing"
    );
    assert_eq!(t.pad_id(), None);
    assert_eq!(t.unk_id(), None);
    assert_eq!(t.bos_id(), None);
    assert!(!t.encode("<|endoftext|>").contains(&t.eos_id().unwrap()));
}

#[test]
fn streaming_reader_and_file_match_in_memory_training() {
    let corpus = format!(
        "{}\n\n{}\n\n{}",
        docs_corpus(),
        "the quick brown fox jumps over the lazy dog.\n".repeat(1600),
        "another line\n\n<|endoftext|>and another doc"
    );
    let mem = ByteBpeTokenizer::train_with_docs(&corpus, 290).unwrap();
    let rdr =
        ByteBpeTokenizer::train_from_reader(std::io::Cursor::new(corpus.as_bytes()), 290).unwrap();
    assert_eq!(mem.encode_state(), rdr.encode_state());

    let docs: Vec<&str> = corpus.split(DOC_SEP).collect();
    let from_docs = ByteBpeTokenizer::train_from_docs(docs.iter().copied().map(Ok), 290).unwrap();
    assert_eq!(mem.encode_state(), from_docs.encode_state());

    let path = std::env::temp_dir().join(format!("ferrum_bpe_stream_{}.txt", std::process::id()));
    std::fs::write(&path, &corpus).unwrap();
    let file_tok = ByteBpeTokenizer::train_from_file(&path, 290).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(mem.encode_state(), file_tok.encode_state());

    assert!(ByteBpeTokenizer::train_from_reader(std::io::Cursor::new(b"abc"), 256).is_err());
    assert!(ByteBpeTokenizer::train_from_reader(std::io::Cursor::new(b""), 300).is_ok());
    assert!(ByteBpeTokenizer::train_from_docs(std::iter::once(Ok("abc")), 256).is_err());
    assert!(
        ByteBpeTokenizer::train_from_docs(vec![Err(InferError::Format("boom".into()))], 300)
            .is_err()
    );
    assert!(ByteBpeTokenizer::train_from_file(
        std::path::Path::new("/nonexistent/ferrum/bpe.txt"),
        300
    )
    .is_err());
}
