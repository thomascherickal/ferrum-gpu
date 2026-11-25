use super::*;

const FRAGMENTS_UNDER_TEST: &[(&str, &str)] = &[
    ("common/leaf", "fn {{NAME}}() -> f32 { return {{VALUE}}; }"),
    (
        "common/outer",
        "//! use common/leaf NAME=one VALUE=1.0\nconst TWO = 2.0;",
    ),
    ("common/loop_a", "//! use common/loop_b"),
    ("common/loop_b", "//! use common/loop_a"),
];

fn composed(src: &str) -> Result<String, String> {
    compose(src, FRAGMENTS_UNDER_TEST)
}

#[test]
fn marker_lines_expand_recursively_with_their_values() {
    let out = composed("// head\n  //! use common/outer\nfn main() {}").unwrap();
    assert_eq!(
        out,
        "// head\nfn one() -> f32 { return 1.0; }\nconst TWO = 2.0;\nfn main() {}\n"
    );
    assert_eq!(composed("plain").unwrap(), "plain\n");
}

#[test]
fn bad_markers_are_refused_with_a_reason() {
    let unknown = composed("//! use common/missing").unwrap_err();
    assert!(unknown.contains("unknown WGSL fragment `common/missing`"));
    assert!(composed("//! use common/loop_a")
        .unwrap_err()
        .contains("cycle"));
    let pair = composed("//! use common/leaf NAME").unwrap_err();
    assert!(pair.contains("`NAME` is not KEY=value"));
    let unset = composed("//! use common/leaf NAME=x").unwrap_err();
    assert!(unset.contains("placeholder is left unset"));
}
