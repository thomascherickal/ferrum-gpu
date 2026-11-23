use super::*;
use ferrum_test_support::link;

#[test]
fn expand_inputs_walks_skips_hidden_and_own_outputs() {
    let d = tmpdir("walk");
    let out = d.join("out");
    std::fs::create_dir_all(out.join("nested")).unwrap();
    std::fs::write(d.join("a.txt"), b"a").unwrap();
    std::fs::write(d.join(".hidden"), b"h").unwrap();
    std::fs::write(out.join("train.txt"), b"t").unwrap();
    let dir = d.to_string_lossy().to_string();
    assert_eq!(
        expand_inputs(std::slice::from_ref(&dir), &out).unwrap(),
        vec![d.join("a.txt")]
    );
    assert!(expand_inputs(&[out.to_string_lossy().into()], &out)
        .unwrap()
        .is_empty());
    let e = expand_inputs(&[out.join("train.txt").to_string_lossy().into()], &out).unwrap_err();
    assert!(e.contains("inside --out"), "{e}");
    let e = expand_inputs(&["/nonexistent-ferrum".into()], &out).unwrap_err();
    assert!(e.contains("does not exist"), "{e}");
    // Not a regular file: refused when named (Unix only: no Windows path is neither file nor folder).
    #[cfg(unix)]
    {
        let e = expand_inputs(&["/dev/null".into()], &out).unwrap_err();
        assert!(e.contains("not a regular file"), "{e}");
    }
    // a link to nothing is skipped inside a folder
    if link(&d.join("missing"), &d.join("dangling")) {
        assert_eq!(expand_inputs(&[dir], &out).unwrap(), vec![d.join("a.txt")]);
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_folder_that_cannot_be_listed_is_named() {
    let d = tmpdir("readdir");
    let refuse = |_: &Path| Err(std::io::Error::other("denied"));
    let e = expand_inputs_with(&[d.to_string_lossy().into()], &d.join("out"), &refuse);
    assert_eq!(
        e.unwrap_err(),
        format!("cannot read {}: denied", d.display())
    );
    let _ = std::fs::remove_dir_all(&d);
}
