use super::*;

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

#[test]
fn an_old_manifest_that_cannot_be_removed_stops_the_run_and_leaves_no_part_files() {
    let d = tmp("nomanifest");
    let f = d.join("c.txt");
    std::fs::write(
        &f,
        "a doc here<|endoftext|>another doc here<|endoftext|>a third document",
    )
    .unwrap();
    let out = d.join("out");
    // a directory where manifest.json should be: it cannot be removed as a file
    std::fs::create_dir_all(out.join("manifest.json")).unwrap();
    let e = prepare(&opts(&f, &out)).unwrap_err();
    assert!(
        e.starts_with("cannot remove ") && e.contains("manifest.json"),
        "{e}"
    );
    assert!(!out.join("train.txt.part").exists());
    let _ = std::fs::remove_dir_all(&d);
}
