use super::support::{put, tmp};
use crate::fsapi::preview::preview;
use crate::fsapi::MAX_PREVIEW;

#[test]
fn a_short_file_is_returned_whole_with_its_line_count() {
    let d = tmp();
    let f = put(&d, "t.txt", b"one\ntwo\nthree\n");
    let p = preview(&f, 1000).unwrap();
    assert_eq!(p.text, "one\ntwo\nthree\n");
    assert_eq!(
        (p.size, p.lines, p.truncated, p.binary),
        (14, 3, false, false)
    );
}

#[test]
fn a_long_file_is_cut_but_its_lines_are_still_counted() {
    let d = tmp();
    let f = put(&d, "t.txt", "line\n".repeat(1000).as_bytes());
    let p = preview(&f, 12).unwrap();
    assert_eq!(p.text, "line\nline\nli");
    assert!(p.truncated);
    assert_eq!(p.size, 5000);
    assert_eq!(p.lines, 1000, "counted past the preview");
}

#[test]
fn the_preview_never_exceeds_the_limit() {
    let d = tmp();
    let f = put(&d, "big.txt", &vec![b'a'; MAX_PREVIEW + 100]);
    let p = preview(&f, usize::MAX).unwrap();
    assert_eq!(p.text.len(), MAX_PREVIEW);
    assert!(p.truncated);
    // a request for zero bytes still returns one
    assert_eq!(preview(&f, 0).unwrap().text.len(), 1);
}

#[test]
fn binary_and_invalid_utf8_files_are_flagged_and_shown_safely() {
    let d = tmp();
    let bin = put(&d, "m.gguf", b"GGUF\0\0\0\x03rest");
    assert!(preview(&bin, 100).unwrap().binary);
    let bad = put(&d, "bad.txt", &[b'o', b'k', 0xff, 0xfe, b'!']);
    let p = preview(&bad, 100).unwrap();
    assert!(!p.binary);
    assert!(p.text.contains('\u{fffd}') && p.text.starts_with("ok"));
}

#[test]
fn an_empty_file_has_an_empty_preview() {
    let d = tmp();
    let f = put(&d, "e.txt", b"");
    let p = preview(&f, 100).unwrap();
    assert_eq!(
        (p.text.as_str(), p.size, p.lines, p.truncated),
        ("", 0, 0, false)
    );
}

#[test]
fn folders_and_missing_files_are_errors_that_name_the_path() {
    let d = tmp();
    assert!(preview(&d, 10).unwrap_err().contains("not a file"));
    let e = preview(&d.join("nope.txt"), 10).unwrap_err();
    assert!(e.contains("nope.txt"), "{e}");
}
