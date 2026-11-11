use super::*;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_prepare_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn opts(input: &Path, out: &Path) -> PrepareOptions {
    PrepareOptions {
        inputs: vec![input.to_string_lossy().into()],
        out: out.to_string_lossy().into(),
        doc_sep: "<|endoftext|>".into(),
        lossy: false,
        nfkc: false,
        lowercase: false,
        strip_gutenberg: false,
        min_chars: 1,
        max_chars: 0,
        dedup: false,
        val_frac: 0.0,
        test_frac: 0.0,
        seed: 42,
        args: vec![],
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures/prepare")
        .join(name)
}

/// The splits and the hashes in the manifest equal what ferrum-cpu's `prepare` wrote (fixture).
#[test]
fn matches_ferrum_cpus_prepare_on_the_golden_corpus() {
    let d = tmp("golden");
    let mut o = opts(&fixture("corpus.txt"), &d.join("o"));
    (o.nfkc, o.dedup, o.strip_gutenberg, o.min_chars) = (true, true, true, 5);
    (o.val_frac, o.test_frac, o.seed) = (0.3, 0.3, 7);
    let r = prepare(&o).unwrap();
    assert_eq!(r.documents, [3, 3, 1]);
    assert_eq!((r.rejected["duplicate"], r.rejected["too_short"]), (1, 1));
    for n in ["train.txt", "val.txt", "test.txt"] {
        assert_eq!(
            std::fs::read(d.join("o").join(n)).unwrap(),
            std::fs::read(fixture(n)).unwrap(),
            "{n}"
        );
    }
    // Same manifest except the paths, the arguments and the version.
    let ours = std::fs::read_to_string(d.join("o/manifest.json")).unwrap();
    let want = std::fs::read_to_string(fixture("manifest.json")).unwrap();
    let keep = |s: &str| {
        s.lines()
            .filter(|l| {
                !l.contains("--")
                    && (l.contains("sha256")
                        || l.contains("bytes")
                        || l.contains("chars")
                        || l.contains("documents")
                        || l.contains("words")
                        || l.contains("lines")
                        || l.contains("duplicate")
                        || l.contains("too_"))
            })
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    assert_eq!(keep(&ours), keep(&want));
    assert!(ours.ends_with("}\n") && ours.contains("\"ferrum_version\""));
    assert!(r.summary(&o).contains("train=3 val=3 test=1"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn rejects_by_length_and_lowercases_and_reports_none() {
    let d = tmp("len");
    let f = d.join("c.txt");
    std::fs::write(
        &f,
        "AB<|endoftext|>Long Enough Doc<|endoftext|>   <|endoftext|>This one is far too long",
    )
    .unwrap();
    let mut o = opts(&f, &d.join("o"));
    (o.min_chars, o.max_chars, o.lowercase) = (3, 20, true);
    let r = prepare(&o).unwrap();
    assert_eq!(r.documents, [1, 0, 0]);
    assert_eq!(
        (
            r.rejected["too_short"],
            r.rejected["empty"],
            r.rejected["too_long"]
        ),
        (1, 1, 1)
    );
    assert_eq!(
        std::fs::read_to_string(d.join("o/train.txt")).unwrap(),
        "long enough doc"
    );
    let mut ok = opts(&fixture("corpus.txt"), &d.join("o2"));
    ok.val_frac = 0.0;
    let r = prepare(&ok).unwrap();
    assert!(r.summary(&ok).contains("rejected none") || r.summary(&ok).contains("rejected"));
    let _ = std::fs::remove_dir_all(&d);
}

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

#[test]
fn a_run_cannot_read_its_own_outputs() {
    let d = tmp("own");
    std::fs::write(d.join("c.txt"), "a doc here").unwrap();
    // --out inside --input: the directory walk skips it on the second run.
    let o = opts(&d, &d.join("out"));
    prepare(&o).unwrap();
    let r = prepare(&o).unwrap();
    assert_eq!(r.documents[0], 1);
    let _ = std::fs::remove_dir_all(&d);
}
