use super::*;

#[test]
fn a_missing_count_is_named() {
    let e = err(&Spec {
        skip_docs: true,
        ..Spec::default()
    });
    assert!(
        e.ends_with("key 'ferrum.tokens.document_count' missing"),
        "{e}"
    );
}
