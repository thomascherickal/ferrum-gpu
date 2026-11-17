//! The accessibility contract of the page markup, checked without a browser: every form control has
//! a name (a wrapping label, a `for` label or an `aria-label`), every button has text or a label,
//! the tabs and panels point at each other, table headers say what they head, ids are unique, and
//! the page declares its language. Dynamic content (the job log, charts) is checked in the browser
//! tests; this is what a static scan can promise.

use std::collections::BTreeSet;
use std::fs;

const DIR: &str = env!("CARGO_MANIFEST_DIR");

fn page() -> String {
    fs::read_to_string(format!("{DIR}/web/index.html")).expect("web/index.html")
}

/// The value of attribute `name` in a tag's text, if present.
fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let head = tag.split('>').next()?;
    let key = format!("{name}=\"");
    // a longer attribute name that ends in `name` (`data-id`) is not this attribute: take the
    // first occurrence that starts a name
    head.match_indices(&key)
        .find(|(i, _)| !head[..*i].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '-'))
        .and_then(|(i, _)| head[i + key.len()..].split('"').next())
}

fn name_of(tag: &str) -> &str {
    let end = tag
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '/'))
        .unwrap_or(tag.len());
    &tag[..end]
}

/// `(tag text from after '<', text up to the next '<')` for every tag, in order.
fn tags(html: &str) -> Vec<(&str, &str)> {
    html.split('<')
        .skip(1)
        .map(|seg| {
            let after = seg.split_once('>').map_or("", |x| x.1);
            (seg, after)
        })
        .collect()
}

#[test]
fn the_page_declares_its_language_and_has_a_title() {
    let h = page();
    assert!(h.contains("<html lang=\""), "<html> needs a lang attribute");
    assert!(h.contains("<title>") && h.contains("name=\"viewport\""));
}

#[test]
fn every_id_is_unique() {
    let h = page();
    let mut seen = BTreeSet::new();
    for (tag, _) in tags(&h) {
        if let Some(id) = attr(tag, "id") {
            assert!(seen.insert(id.to_string()), "id {id:?} appears twice");
        }
    }
    assert!(
        seen.len() > 100,
        "the scan should see the whole page ({})",
        seen.len()
    );
}

#[test]
fn every_form_control_has_a_name() {
    let h = page();
    let for_ids: BTreeSet<&str> = tags(&h)
        .iter()
        .filter(|(t, _)| name_of(t) == "label")
        .filter_map(|(t, _)| attr(t, "for"))
        .collect();
    let mut open_labels = 0i32;
    let mut checked = 0;
    for (tag, _) in tags(&h) {
        match name_of(tag) {
            "label" => open_labels += 1,
            "/label" => open_labels -= 1,
            "input" | "select" | "textarea" => {
                if attr(tag, "type") == Some("hidden") {
                    continue;
                }
                checked += 1;
                let named = open_labels > 0
                    || attr(tag, "aria-label").is_some()
                    || attr(tag, "aria-labelledby").is_some()
                    || attr(tag, "id").is_some_and(|id| for_ids.contains(id));
                assert!(
                    named,
                    "a control has no label: <{}",
                    tag.split('>').next().unwrap()
                );
            }
            _ => {}
        }
    }
    assert!(checked > 80, "{checked} controls scanned");
}

#[test]
fn every_button_has_visible_text_or_a_label() {
    let h = page();
    let mut checked = 0;
    for (tag, after) in tags(&h) {
        if name_of(tag) != "button" {
            continue;
        }
        checked += 1;
        let text = after.trim();
        assert!(
            !text.is_empty() || attr(tag, "aria-label").is_some(),
            "a button has no text: <{}",
            tag.split('>').next().unwrap()
        );
    }
    assert!(checked > 30, "{checked} buttons scanned");
}

#[test]
fn tabs_and_panels_point_at_each_other() {
    let h = page();
    let all = tags(&h);
    let ids: BTreeSet<&str> = all.iter().filter_map(|(t, _)| attr(t, "id")).collect();
    let (mut tabs, mut panels) = (0, 0);
    for (tag, _) in &all {
        match attr(tag, "role") {
            Some("tab") => {
                tabs += 1;
                let id = attr(tag, "id").expect("a tab needs an id");
                let target =
                    attr(tag, "aria-controls").unwrap_or_else(|| panic!("{id}: aria-controls"));
                assert!(
                    ids.contains(target),
                    "{id} controls {target}, which does not exist"
                );
                assert!(attr(tag, "aria-selected").is_some(), "{id}: aria-selected");
            }
            Some("tabpanel") => {
                panels += 1;
                let id = attr(tag, "id").expect("a panel needs an id");
                let by =
                    attr(tag, "aria-labelledby").unwrap_or_else(|| panic!("{id}: aria-labelledby"));
                assert!(
                    ids.contains(by),
                    "{id} is labelled by {by}, which does not exist"
                );
            }
            _ => {}
        }
    }
    assert!(tabs >= 12 && tabs == panels, "{tabs} tabs, {panels} panels");
    assert!(h.contains("role=\"tablist\""), "the tabs need a tablist");
}

#[test]
fn table_headers_say_which_way_they_head() {
    let h = page();
    let mut checked = 0;
    for (tag, _) in tags(&h) {
        if name_of(tag) == "th" {
            checked += 1;
            assert!(
                matches!(attr(tag, "scope"), Some("col" | "row")),
                "a <th> needs scope: <{}",
                tag.split('>').next().unwrap()
            );
        }
    }
    assert!(checked > 20, "{checked} headers scanned");
}

#[test]
fn images_have_alternative_text_and_status_text_is_announced() {
    let h = page();
    for (tag, _) in tags(&h) {
        if name_of(tag) == "img" {
            assert!(attr(tag, "alt").is_some(), "an <img> needs alt text");
        }
    }
    // the notes that change after an action are live regions
    for id in ["cap-note", "gl-summary", "cap-ceiling"] {
        let tag = tags(&h)
            .into_iter()
            .map(|t| t.0)
            .find(|t| attr(t, "id") == Some(id))
            .unwrap_or_else(|| panic!("{id} missing"));
        assert!(
            attr(tag, "aria-live").is_some() || attr(tag, "role") == Some("status"),
            "{id} should be a live region"
        );
    }
}

#[test]
fn the_scanner_itself_reads_attributes_correctly() {
    assert_eq!(attr("input id=\"a\" data-id=\"b\">", "id"), Some("a"));
    assert_eq!(attr("input data-id=\"b\" id=\"a\">", "id"), Some("a"));
    assert_eq!(attr("input data-id=\"b\">", "id"), None);
    assert_eq!(attr("button>Go", "aria-label"), None);
    assert_eq!(name_of("/label>"), "/label");
    assert_eq!(name_of("th scope=\"col\">"), "th");
    assert_eq!(tags("<p>x</p>").len(), 2);
}
