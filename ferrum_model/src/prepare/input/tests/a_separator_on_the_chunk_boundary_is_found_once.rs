use super::*;

#[test]
fn a_separator_on_the_chunk_boundary_is_found_once() {
    let sep = "<|endoftext|>";
    let d = tmpdir("straddle");
    let p = d.join("f.txt");
    for back in 1..sep.len() {
        let mut data = vec![b'b'; CHUNK - back];
        data.extend_from_slice(sep.as_bytes());
        data.extend_from_slice(b"tail");
        std::fs::write(&p, &data).unwrap();
        let (docs, offs, hash, total) = split(&p, sep);
        assert_eq!(docs.len(), 2, "back={back}");
        assert_eq!(
            (docs[0].len(), docs[1].as_slice()),
            (CHUNK - back, &b"tail"[..])
        );
        assert_eq!(offs[1], (CHUNK - back + sep.len()) as u64);
        assert_eq!(
            (total, hash),
            (data.len() as u64, crate::sha256::sha256_hex(&data))
        );
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn no_separator_means_one_document_and_an_empty_file_one_empty_document() {
    let d = tmpdir("one");
    let p = d.join("f.txt");
    std::fs::write(&p, vec![b'x'; 2 * CHUNK + 17]).unwrap();
    let (docs, offs, _, total) = split(&p, "");
    assert_eq!(
        (docs.len(), docs[0].len(), offs[0], total),
        (1, 2 * CHUNK + 17, 0, total)
    );
    std::fs::write(&p, b"").unwrap();
    assert_eq!(split(&p, "<|endoftext|>").0, vec![Vec::<u8>::new()]);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn callback_errors_and_missing_files_propagate() {
    let d = tmpdir("err");
    let p = d.join("f.txt");
    std::fs::write(&p, b"a|b").unwrap();
    let (mut h, mut t) = (Sha256::new(), 0);
    let e = for_each_raw_doc(&p, "|", &mut h, &mut t, |_, _| Err("stop".into())).unwrap_err();
    assert_eq!(e, "stop");
    let e = for_each_raw_doc(&d.join("nope"), "", &mut h, &mut t, |_, _| Ok(())).unwrap_err();
    assert!(e.contains("cannot open"), "{e}");
    // A directory opens on Linux but cannot be read: a clean error.
    assert!(for_each_raw_doc(&d, "", &mut h, &mut t, |_, _| Ok(())).is_err());
    let _ = std::fs::remove_dir_all(&d);
}
