use super::*;
use crate::sha256::sha256_hex;
use crate::tokenizer_gguf::from_gguf_bytes;
use ferrum_ref::trainer::TokenSource;
use std::path::PathBuf;

const TOK: &[u8] = include_bytes!("../../../fixtures/bpe/tok.gguf");
const CORPUS: &str = include_str!("../../../fixtures/bpe/corpus.txt");
const THEIRS: &[u8] = include_bytes!("../../../fixtures/bpe/corpus.tok.gguf");

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("ferrum_tokfile_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).display().to_string()
}

fn meta(vocab: usize, eos: Option<u32>) -> TokenFileMeta {
    TokenFileMeta {
        vocab_size: vocab,
        eos_id: eos,
        tokenizer_sha256: "ab".repeat(32),
    }
}

fn ids(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 31 + 7) % vocab) as u32).collect()
}

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

#[test]
fn an_empty_file_has_no_documents_and_no_eos_means_one_document_per_file() {
    let path = tmp("empty.tok.gguf");
    let s = write(&path, &meta(10, None), |_| Ok(())).unwrap();
    assert_eq!((s.tokens, s.documents), (0, 0));
    let f = TokenFile::open(&path).unwrap();
    assert!(TokenSource::is_empty(&f) && f.eos_id().is_none());
    assert_eq!(f.read(0, 0).unwrap(), Vec::<usize>::new());
    let p2 = tmp("noeos.tok.gguf");
    assert_eq!(
        write(&p2, &meta(10, None), |s| s.push(&[1, 2, 3]))
            .unwrap()
            .documents,
        1
    );
}

#[test]
fn a_failed_write_leaves_the_old_file_and_no_part_file() {
    let path = tmp("atomic.tok.gguf");
    write(&path, &meta(10, None), |s| s.push(&[1, 2, 3])).unwrap();
    let before = std::fs::read(&path).unwrap();
    let e = write(&path, &meta(10, None), |s| s.push(&[1, 99])).unwrap_err();
    assert!(e.contains("outside the vocabulary of 10"), "{e}");
    let e = write(&path, &meta(10, None), |s| {
        s.push(&[4])?;
        Err("boom".into())
    })
    .unwrap_err();
    assert_eq!(e, "boom");
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(!PathBuf::from(format!("{path}.part")).exists());
    assert!(write(&path, &meta(0, None), |_| Ok(()))
        .unwrap_err()
        .contains("vocab_size"));
    assert!(write(&path, &meta(10, Some(10)), |_| Ok(()))
        .unwrap_err()
        .contains("eos_id"));
    assert!(write("/nope/dir/x.gguf", &meta(10, None), |_| Ok(())).is_err());
}

#[test]
fn open_refuses_other_files_and_damaged_ones() {
    let text = tmp("text.txt");
    std::fs::write(&text, "just some text, not a model").unwrap();
    assert!(!is_token_file(&text) && !is_token_file("/nope"));
    assert!(TokenFile::open(&text).is_err());
    assert!(TokenFile::open("/nope").is_err());
    // a tokenizer file is GGUF but not a token file
    let tokf = tmp("tok.gguf");
    std::fs::write(&tokf, TOK).unwrap();
    assert!(is_token_file(&tokf));
    assert!(TokenFile::open(&tokf)
        .unwrap_err()
        .contains("is not a ferrum-tokens file"));
    // truncated: header claims 942 ids but the data stops early
    let cut = tmp("cut.tok.gguf");
    std::fs::write(&cut, &THEIRS[..THEIRS.len() - 10]).unwrap();
    assert!(TokenFile::open(&cut).unwrap_err().contains("extends past"));
    // an id outside the vocabulary is caught when read
    let mut bad = THEIRS.to_vec();
    let at = bad.len() - 2;
    bad[at..].copy_from_slice(&(400i16).to_le_bytes());
    let badp = tmp("badid.tok.gguf");
    std::fs::write(&badp, &bad).unwrap();
    let f = TokenFile::open(&badp).unwrap();
    assert!(f
        .read(941, 1)
        .unwrap_err()
        .contains("outside the vocabulary of 320"));
    assert!(f.read(0, 941).is_ok());
}

#[test]
fn header_length_does_not_depend_on_the_counts() {
    let m = meta(300, Some(1));
    assert_eq!(
        header_bytes_len(&m, 0, 0),
        header_bytes_len(&m, u64::MAX / 2, u64::MAX / 3)
    );
}

fn header_bytes_len(m: &TokenFileMeta, n: u64, d: u64) -> usize {
    super::writer::header_bytes(m, n, d).len()
}
