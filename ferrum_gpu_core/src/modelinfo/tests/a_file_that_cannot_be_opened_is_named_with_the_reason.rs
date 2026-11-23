use super::inspect_with::inspect_via;
use super::*;
use ferrum_test_support::{denied, DiesAfter};

#[test]
fn a_file_that_cannot_be_opened_or_read_is_named_with_the_reason() {
    let f = write(&tmp(), "m.gguf", &model_bytes());
    let e = inspect_via(&f, 100, &denied).unwrap_err();
    assert_eq!(e, format!("{}: denied", f.display()));
    let e = inspect_via(&f, 100, &|_| Ok(Box::new(DiesAfter(b"GG")))).unwrap_err();
    assert_eq!(e, format!("{}: disk gone", f.display()));
}
