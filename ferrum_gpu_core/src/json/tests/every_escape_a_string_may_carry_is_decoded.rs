use super::*;

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
