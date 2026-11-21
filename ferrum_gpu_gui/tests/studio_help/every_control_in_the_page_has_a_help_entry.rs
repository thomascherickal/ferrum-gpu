use super::*;

#[test]
fn every_control_in_the_page_has_a_help_entry() {
    let ids = control_ids(&read("web/index.html"));
    assert!(ids.len() > 25, "found only {} controls", ids.len());
    let help = help_entries(&help_source());
    let missing: Vec<&String> = ids.iter().filter(|i| !help.contains_key(*i)).collect();
    assert!(missing.is_empty(), "controls without help: {missing:?}");
}

#[test]
fn no_help_entry_describes_a_control_that_is_not_in_the_page() {
    let ids = control_ids(&read("web/index.html"));
    let help = help_entries(&help_source());
    let stale: Vec<&String> = help.keys().filter(|k| !ids.contains(k)).collect();
    assert!(
        stale.is_empty(),
        "help for controls that do not exist: {stale:?}"
    );
}

#[test]
fn each_help_text_is_a_real_explanation() {
    for (id, (expert, beginner)) in help_entries(&help_source()) {
        assert!(
            words(&expert) >= 25,
            "{id}: expert text has {} words",
            words(&expert)
        );
        assert!(
            words(&beginner) >= 15,
            "{id}: beginner text has {} words",
            words(&beginner)
        );
        assert_ne!(expert, beginner, "{id}: the two texts must differ");
    }
}

#[test]
fn the_parsers_see_what_they_should() {
    let html = r#"<button id="b1">x</button><input id="i1"><a id="dl" class="button"></a><a id="plain"></a><p id="p"></p><select id="s1"></select><textarea id="t1"></textarea><button>no id</button>"#;
    assert_eq!(control_ids(html), ["b1", "i1", "dl", "s1", "t1"]);
    let js = "  seed: ['one two', 'three four'],\n  'tab-x': ['a b c', 'd e f'],\n// comment\nexport const X = 1;\n";
    let m = help_entries(js);
    assert_eq!(m["seed"], ("one two".into(), "three four".into()));
    assert_eq!(m["tab-x"], ("a b c".into(), "d e f".into()));
    assert_eq!(m.len(), 2);
}

#[test]
fn every_shell_module_is_embedded_and_served() {
    for name in ["index", "tabs", "settings", "tooltip", "help"] {
        let path = format!("/web/shell/{name}.js");
        assert!(
            MODULES.iter().any(|(n, _)| format!("/web/{n}") == path),
            "{path} is not embedded"
        );
        let r = serve(&path).unwrap_or_else(|| panic!("{path} is not served"));
        assert_eq!(r.status, 200);
        assert!(r.content_type.starts_with("text/javascript"));
        assert!(!r.body.is_empty());
    }
    assert!(serve("/web/shell/nope.js").is_none());
}
