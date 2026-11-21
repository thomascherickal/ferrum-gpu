use super::*;

#[test]
fn atomic_write_replaces_the_target_and_leaves_no_temp_file() {
    let d = tmpdir("atomic");
    let p = d.join("run.ckpt");
    atomic_write(&p, b"first").unwrap();
    atomic_write(&p, b"second, longer").unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), b"second, longer");
    let names: Vec<_> = std::fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "no stray temp file: {names:?}");
    assert!(atomic_write(&d.join("no/such/dir/x"), b"x").is_err());
    assert!(atomic_write(Path::new("/"), b"x").is_err());
    let _ = std::fs::remove_dir_all(&d);
}
