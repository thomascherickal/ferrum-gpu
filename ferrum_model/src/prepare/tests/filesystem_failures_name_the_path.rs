use super::*;

fn corpus(tag: &str) -> (PathBuf, PathBuf) {
    let d = tmp(tag);
    let f = d.join("c.txt");
    std::fs::write(&f, "a doc here<|endoftext|>another doc here").unwrap();
    (d, f)
}

#[test]
fn an_output_directory_under_a_regular_file_cannot_be_created() {
    let (d, f) = corpus("mkdir");
    let e = prepare(&opts(&f, &f.join("sub"))).unwrap_err();
    assert!(e.starts_with("cannot create directory "), "{e}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_manifest_that_cannot_be_written_is_named() {
    let (d, f) = corpus("mpart");
    let out = d.join("out");
    std::fs::create_dir_all(out.join("manifest.json.part")).unwrap();
    let e = prepare(&opts(&f, &out)).unwrap_err();
    assert!(
        e.starts_with("cannot write ") && e.contains("manifest.json.part"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn an_output_that_cannot_be_renamed_into_place_is_named() {
    let (d, f) = corpus("rename");
    let out = d.join("out");
    std::fs::create_dir_all(out.join("train.txt")).unwrap();
    let e = prepare(&opts(&f, &out)).unwrap_err();
    assert!(
        e.starts_with("cannot rename ") && e.contains("train.txt"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&d);
}
