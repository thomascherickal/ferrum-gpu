use super::support::{put, tmp};
use crate::fsapi::preview::preview_with;
use ferrum_test_support::{denied, DiesAfter};

#[test]
fn a_file_that_cannot_be_opened_is_named_with_the_reason() {
    let f = put(&tmp(), "t.txt", b"x");
    let e = preview_with(&f, 100, &denied).unwrap_err();
    assert_eq!(e, format!("{}: denied", f.display()));
}

#[test]
fn a_read_that_fails_is_an_error_that_names_the_path() {
    let f = put(&tmp(), "t.txt", b"x");
    let e = preview_with(&f, 100, &|_| Ok(Box::new(DiesAfter(b"ab")))).unwrap_err();
    assert_eq!(e, format!("{}: disk gone", f.display()));
}

#[test]
fn a_failure_while_counting_lines_ends_the_count_not_the_preview() {
    let f = put(&tmp(), "t.txt", b"x");
    let p = preview_with(&f, 3, &|_| Ok(Box::new(DiesAfter(b"a\nb\nc\nd")))).unwrap();
    assert_eq!((p.text.as_str(), p.lines), ("a\nb", 3));
}
