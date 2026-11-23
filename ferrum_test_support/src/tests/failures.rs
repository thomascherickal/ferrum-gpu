use super::*;
use std::io::Write;

#[test]
fn a_reader_serves_its_data_then_fails() {
    let mut r = DiesAfter(b"abc");
    let mut buf = [0u8; 2];
    assert_eq!(r.read(&mut buf).unwrap(), 2);
    assert_eq!(r.read(&mut buf).unwrap(), 1);
    assert_eq!(r.read(&mut buf).unwrap_err().to_string(), "disk gone");
}

#[test]
fn a_refused_open_and_a_broken_writer_say_why() {
    let e = denied(Path::new("x")).err().unwrap();
    assert_eq!(e.kind(), io::ErrorKind::PermissionDenied);
    assert!(BrokenWriter::Write.write(b"x").is_err());
    assert_eq!(BrokenWriter::Flush.write(b"xy").unwrap(), 2);
    assert!(BrokenWriter::Flush.flush().is_err());
}

#[test]
fn a_link_is_made_where_the_platform_allows_it() {
    let dir = std::env::temp_dir().join(format!("ferrum_link_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.join("t.txt");
    std::fs::write(&target, b"x").unwrap();
    if link(&target, &dir.join("l.txt")) {
        assert_eq!(std::fs::read(dir.join("l.txt")).unwrap(), b"x");
    }
}
