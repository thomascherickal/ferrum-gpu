use super::*;
use ferrum_test_support::BrokenWriter;

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

#[test]
fn a_failed_write_and_a_failed_flush_name_the_part_file() {
    let broken = |kind| {
        let tmp = PathBuf::from("train.txt.part");
        Split::with_writer("train.txt".into(), tmp, Box::new(kind))
    };
    let mut s = broken(BrokenWriter::Write);
    let e = s.write(&"x".repeat(1 << 20)).unwrap_err();
    assert_eq!(e, "cannot write train.txt.part: no space left");
    let mut s = broken(BrokenWriter::Flush);
    s.write("small").unwrap();
    let e = s.finish().err().unwrap();
    assert_eq!(e, "cannot write train.txt.part: no space left");
}
