use super::*;

fn p(s: &str) -> Result<Json, String> {
    Json::parse(s)
}

#[test]
fn every_escape_a_string_may_carry_is_decoded() {
    assert_eq!(
        p(r#""\"\\\/\b\f\n\r\t\u0041""#),
        Ok(Json::Str("\"\\/\u{8}\u{c}\n\r\tA".into()))
    );
}

#[test]
fn parses_every_value_kind() {
    assert_eq!(p("null"), Ok(Json::Null));
    assert_eq!(p(" true "), Ok(Json::Bool(true)));
    assert_eq!(p("false"), Ok(Json::Bool(false)));
    assert_eq!(p("-12.5e1"), Ok(Json::Num(-125.0)));
    assert_eq!(
        p("\"a\\n\\u00e9\\ud83d\\ude00\""),
        Ok(Json::Str("a\né😀".into()))
    );
    assert_eq!(p("[]"), Ok(Json::Arr(vec![])));
    assert_eq!(p("{}"), Ok(Json::Obj(vec![])));
    let v = p("{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}").unwrap();
    assert_eq!(v.get("c").and_then(Json::as_str), Some("x"));
    assert_eq!(
        v.get("a"),
        Some(&Json::Arr(vec![
            Json::Num(1.0),
            Json::Num(2.0),
            Json::obj([("b", Json::Null)])
        ]))
    );
    assert_eq!(v.get("missing"), None);
    assert_eq!(Json::Null.get("a"), None);
}

#[test]
fn display_round_trips_and_escapes() {
    let v = Json::obj([
        ("s", Json::Str("q\"\\\n\t\r\u{1}é".into())),
        (
            "a",
            Json::Arr(vec![Json::Num(1.5), Json::Bool(false), Json::Null]),
        ),
        ("n", Json::Num(-3.0)),
    ]);
    assert_eq!(p(&v.to_string()), Ok(v.clone()));
    assert!(v.to_string().contains("\\u0001"));
    assert_eq!(Json::Num(f64::NAN).to_string(), "null");
    assert_eq!(Json::Num(f64::INFINITY).to_string(), "null");
    assert_eq!(Json::Num(7.0).to_string(), "7");
}

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
