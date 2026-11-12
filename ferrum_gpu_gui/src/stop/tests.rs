use super::*;

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

fn req(text: &str) -> Json {
    Json::parse(text).unwrap()
}

#[test]
fn text_is_cut_before_the_earliest_stop() {
    assert_eq!(
        cut_at_stop("abc. def. ghi".into(), &s(&["def", "."])),
        "abc"
    );
    assert_eq!(cut_at_stop("abc. def".into(), &s(&["zzz", "def"])), "abc. ");
}

#[test]
fn no_stops_or_no_match_leaves_the_text_alone() {
    assert_eq!(cut_at_stop("abc".into(), &[]), "abc");
    assert_eq!(cut_at_stop("abc".into(), &s(&["x"])), "abc");
}

#[test]
fn an_empty_stop_string_never_matches() {
    assert_eq!(cut_at_stop("abc".into(), &s(&[""])), "abc");
    assert_eq!(cut_at_stop("abc".into(), &s(&["", "c"])), "ab");
}

#[test]
fn a_stop_at_the_very_start_empties_the_text() {
    assert_eq!(cut_at_stop("abc".into(), &s(&["a"])), "");
}

#[test]
fn a_missing_stop_member_means_none() {
    assert_eq!(
        parse(&req(r#"{"prompt":"x"}"#)).unwrap(),
        Vec::<String>::new()
    );
}

#[test]
fn an_array_of_strings_is_read_in_order() {
    assert_eq!(
        parse(&req(r#"{"stop":["a","\n"]}"#)).unwrap(),
        s(&["a", "\n"])
    );
}

#[test]
fn the_wrong_shapes_are_refused() {
    for bad in [r#"{"stop":"a"}"#, r#"{"stop":[1]}"#, r#"{"stop":{"a":1}}"#] {
        let e = parse(&req(bad)).unwrap_err();
        assert!(e.contains("array of strings"), "{bad}: {e}");
    }
}

#[test]
fn the_count_and_length_limits_are_enforced() {
    let many = format!(r#"{{"stop":[{}]}}"#, [r#""a""#; MAX_STOPS + 1].join(","));
    assert!(parse(&req(&many)).unwrap_err().contains("at most"));
    let ok = format!(r#"{{"stop":[{}]}}"#, [r#""a""#; MAX_STOPS].join(","));
    assert_eq!(parse(&req(&ok)).unwrap().len(), MAX_STOPS);
    let long = format!(r#"{{"stop":["{}"]}}"#, "x".repeat(MAX_STOP_LEN + 1));
    assert!(parse(&req(&long)).unwrap_err().contains("over"));
    let edge = format!(r#"{{"stop":["{}"]}}"#, "x".repeat(MAX_STOP_LEN));
    assert!(parse(&req(&edge)).is_ok());
}
