use super::*;

#[test]
fn malformed_documents_are_errors_not_panics() {
    for bad in [
        "",
        "   ",
        "nul",
        "tru",
        "{",
        "[",
        "[1,",
        "[1 2]",
        "{\"a\"}",
        "{\"a\":}",
        "{1:2}",
        "{\"a\":1,}",
        "\"abc",
        "\"\\x\"",
        "\"\\u12\"",
        "\"\\ud800\"",
        "\"\\ud800\\u0041\"",
        "\"\u{1}\"",
        "1 2",
        "[1]]",
        "-",
        "1e999",
        "--1",
        "@",
        "{\"a\":1 \"b\":2}",
        "\"\\",
        "\"\\ud800\\",
        "[\"a\" \"b\"]",
    ] {
        assert!(p(bad).is_err(), "{bad:?} should be refused");
    }
}

#[test]
fn nesting_is_bounded() {
    let ok = format!("{}1{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
    assert!(p(&ok).is_ok());
    let deep = format!(
        "{}1{}",
        "[".repeat(MAX_DEPTH + 2),
        "]".repeat(MAX_DEPTH + 2)
    );
    assert!(p(&deep).unwrap_err().contains("deep"));
    let huge = "[".repeat(100_000);
    assert!(p(&huge).is_err(), "no stack overflow on hostile depth");
}

#[test]
fn typed_accessors() {
    assert_eq!(Json::Num(3.0).as_usize(), Some(3));
    assert_eq!(Json::Num(3.5).as_usize(), None);
    assert_eq!(Json::Num(-1.0).as_usize(), None);
    assert_eq!(Json::Num(1e300).as_usize(), None);
    assert_eq!(Json::Str("x".into()).as_usize(), None);
    assert_eq!(Json::Str("x".into()).as_f64(), None);
    assert_eq!(Json::Null.as_str(), None);
    assert_eq!(Json::Num(2.5).as_f64(), Some(2.5));
}
