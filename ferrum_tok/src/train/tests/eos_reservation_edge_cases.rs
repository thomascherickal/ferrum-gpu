use super::*;

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
