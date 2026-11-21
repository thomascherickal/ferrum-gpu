use super::*;

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
    // Not a regular file: refused when named, skipped inside a directory.
    #[cfg(unix)]
    {
        let e = expand_inputs(&["/dev/zero".into()], &out).unwrap_err();
        assert!(e.contains("not a regular file"), "{e}");
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(d.join("missing"), d.join("dangling")).unwrap();
        assert_eq!(expand_inputs(&[dir], &out).unwrap(), vec![d.join("a.txt")]);
    }
    let _ = std::fs::remove_dir_all(&d);
}
