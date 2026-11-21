use super::*;

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
