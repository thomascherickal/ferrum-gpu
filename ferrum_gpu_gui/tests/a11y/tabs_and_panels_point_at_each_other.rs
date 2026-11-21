use super::*;

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
