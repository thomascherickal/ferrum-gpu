use super::*;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_split_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn a_split_joins_documents_and_counts_them() {
    let d = tmp("ok");
    let mut s = Split::create(&d, "train").unwrap();
    s.write("ab").unwrap();
    s.write("cd").unwrap();
    let (json, bytes, chars) = s.finish().unwrap();
    let want = format!("ab{JOIN}cd");
    assert_eq!((bytes as usize, chars), (want.len(), 4));
    assert!(json
        .pretty()
        .contains(&crate::sha256::sha256_hex(want.as_bytes())));
    assert_eq!(
        std::fs::read_to_string(part(&d, "train.txt")).unwrap(),
        want
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn create_names_the_part_file_when_it_cannot_be_made() {
    let e = Split::create(Path::new("/nonexistent-ferrum-dir"), "train")
        .err()
        .unwrap();
    assert!(
        e.starts_with("cannot create ") && e.contains("train.txt.part"),
        "{e}"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_failed_write_and_a_failed_flush_name_the_part_file() {
    let d = tmp("full");
    std::os::unix::fs::symlink("/dev/full", part(&d, "train.txt")).unwrap();
    let mut s = Split::create(&d, "train").unwrap();
    let e = s.write(&"x".repeat(1 << 20)).unwrap_err();
    assert!(
        e.starts_with("cannot write ") && e.contains("train.txt.part"),
        "{e}"
    );
    let d2 = tmp("full2");
    std::os::unix::fs::symlink("/dev/full", part(&d2, "train.txt")).unwrap();
    let mut s = Split::create(&d2, "train").unwrap();
    s.write("small").unwrap();
    assert!(s.finish().err().unwrap().starts_with("cannot write "));
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_dir_all(&d2);
}
