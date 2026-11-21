use super::*;

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
