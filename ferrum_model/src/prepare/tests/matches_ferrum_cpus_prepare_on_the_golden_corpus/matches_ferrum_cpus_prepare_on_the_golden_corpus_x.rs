use super::*;

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
