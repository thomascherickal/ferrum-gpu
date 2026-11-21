use super::*;

#[test]
fn writing_the_fixture_corpus_reproduces_ferrum_cpus_file_byte_for_byte() {
    let tok = from_gguf_bytes(TOK).unwrap();
    let m = TokenFileMeta {
        vocab_size: tok.vocab_size(),
        eos_id: tok.eos_id().map(|e| e as u32),
        tokenizer_sha256: sha256_hex(TOK),
    };
    let path = tmp("golden.tok.gguf");
    let all: Vec<u32> = tok
        .encode_docs(CORPUS)
        .into_iter()
        .map(|i| i as u32)
        .collect();
    let stats = write(&path, &m, |s| s.push(&all)).unwrap();
    assert_eq!((stats.tokens, stats.documents), (942, 31));
    let ours = std::fs::read(&path).unwrap();
    assert_eq!(stats.bytes as usize, ours.len());
    assert_eq!(ours.len(), THEIRS.len());
    assert_eq!(ours, THEIRS, "same bytes as ferrum-cpu's `tokenize` output");
}

#[test]
fn reading_ferrum_cpus_file_gives_its_ids_and_metadata() {
    let path = tmp("theirs.tok.gguf");
    std::fs::write(&path, THEIRS).unwrap();
    assert!(is_token_file(&path));
    let f = TokenFile::open(&path).unwrap();
    assert_eq!(
        (
            TokenSource::len(&f),
            f.vocab_size(),
            f.documents(),
            f.eos_id()
        ),
        (942, 320, 31, Some(319))
    );
    assert_eq!(
        f.tokenizer_sha256(),
        sha256_hex(TOK),
        "the recorded tokenizer digest is the fixture's"
    );
    let tok = from_gguf_bytes(TOK).unwrap();
    assert_eq!(f.read(0, 942).unwrap(), tok.encode_docs(CORPUS));
    assert_eq!(f.window(100, 5).unwrap(), tok.encode_docs(CORPUS)[100..105]);
}

#[test]
fn round_trips_narrow_and_wide_vocabularies() {
    for (vocab, width) in [(300usize, 2usize), (40_000, 4)] {
        let path = tmp(&format!("v{vocab}.tok.gguf"));
        let data = ids(1000, vocab);
        let s = write(&path, &meta(vocab, Some(5)), |sink| {
            assert!(sink.is_empty());
            sink.push(&data[..400])?;
            assert_eq!(sink.len(), 400);
            sink.push(&data[400..])
        })
        .unwrap();
        assert_eq!(s.tokens, 1000);
        assert_eq!(element_type(vocab).1, width);
        let f = TokenFile::open(&path).unwrap();
        let back = f.read(0, 1000).unwrap();
        assert_eq!(back, data.iter().map(|&x| x as usize).collect::<Vec<_>>());
        assert_eq!(f.read(990, 10).unwrap(), back[990..]);
        assert!(f.read(995, 6).unwrap_err().contains("outside"));
        assert!(f.read(usize::MAX, 2).unwrap_err().contains("outside"));
        assert_eq!(
            f.documents(),
            data.iter().filter(|&&x| x == 5).count() as u64 + 1
        );
    }
}
