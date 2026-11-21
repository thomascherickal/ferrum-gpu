use super::*;

#[test]
fn invalid_utf8_fails_with_the_offset_or_is_repaired_with_lossy() {
    let d = tmp("utf8");
    let f = d.join("c.txt");
    std::fs::write(&f, b"good doc<|endoftext|>bad \xff doc").unwrap();
    let e = prepare(&opts(&f, &d.join("o"))).unwrap_err();
    assert!(
        e.contains("invalid UTF-8 at byte 25") && e.contains("--lossy"),
        "{e}"
    );
    assert!(!d.join("o/train.txt.part").exists() && !d.join("o/manifest.json").exists());
    let mut o = opts(&f, &d.join("o"));
    o.lossy = true;
    let r = prepare(&o).unwrap();
    assert_eq!((r.lossy_repaired, r.documents[0]), (1, 2));
    assert!(std::fs::read_to_string(d.join("o/manifest.json"))
        .unwrap()
        .contains("lossy_repaired"));
    assert!(r.summary(&o).contains("lossy_repaired=1"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn bad_fractions_an_empty_train_split_and_a_failed_run_keep_old_outputs() {
    let d = tmp("err");
    let f = d.join("c.txt");
    std::fs::write(&f, "a doc here").unwrap();
    let mut o = opts(&f, &d.join("o"));
    o.val_frac = 0.6;
    o.test_frac = 0.6;
    assert!(prepare(&o).unwrap_err().contains("sum to less than 1"));
    o.val_frac = 1.0; // not in [0, 1)
    assert!(prepare(&o).unwrap_err().contains("[0, 1)"));
    // A good run, then a failing one: the first run's files stay.
    o.val_frac = 0.0;
    o.test_frac = 0.0;
    prepare(&o).unwrap();
    let before = std::fs::read(d.join("o/manifest.json")).unwrap();
    o.min_chars = 1000; // rejects everything: empty train split
    let e = prepare(&o).unwrap_err();
    assert!(e.contains("train split is empty"), "{e}");
    assert_eq!(std::fs::read(d.join("o/manifest.json")).unwrap(), before);
    assert!(std::fs::read_dir(d.join("o")).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".part")));
    o.inputs = vec!["/nonexistent-ferrum".into()];
    assert!(prepare(&o).unwrap_err().contains("does not exist"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn dedup_and_a_directory_of_files_are_deterministic() {
    let d = tmp("dir");
    std::fs::create_dir_all(d.join("in")).unwrap();
    std::fs::write(
        d.join("in/b.txt"),
        "same doc here<|endoftext|>other doc here",
    )
    .unwrap();
    std::fs::write(d.join("in/a.txt"), "same doc here\r\nline two").unwrap();
    let mut o = opts(&d.join("in"), &d.join("o1"));
    o.dedup = true;
    o.val_frac = 0.2;
    o.test_frac = 0.2;
    let r1 = prepare(&o).unwrap();
    o.out = d.join("o2").to_string_lossy().into();
    let r2 = prepare(&o).unwrap();
    assert_eq!(r1.documents, r2.documents);
    for n in ["train.txt", "val.txt", "test.txt"] {
        assert_eq!(
            std::fs::read(d.join("o1").join(n)).unwrap(),
            std::fs::read(d.join("o2").join(n)).unwrap()
        );
    }
    // a.txt is "same doc here\nline two" after CRLF folding, so it is not a duplicate of b's first.
    assert_eq!(r1.rejected["duplicate"], 0);
    let _ = std::fs::remove_dir_all(&d);
}
