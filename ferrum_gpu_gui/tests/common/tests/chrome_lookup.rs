use crate::common::chrome;

/// Looking for a browser must never panic or hang, whether or not one is installed.
#[test]
fn looking_for_a_browser_answers_with_a_name_or_nothing() {
    if let Some(name) = chrome() {
        assert!(!name.is_empty());
    }
}
