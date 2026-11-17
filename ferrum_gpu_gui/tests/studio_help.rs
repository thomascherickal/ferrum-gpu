//! The tool-tip contract, checked without a browser or Node: every control in the page has a help
//! entry, no entry is stale, and each entry is a real explanation (the expert text 25 words or more,
//! the beginner text 15 or more). Also that every shell module is embedded and served.

use ferrum_gpu_gui::assets::{serve, MODULES};
use std::collections::BTreeMap;
use std::fs;

const DIR: &str = env!("CARGO_MANIFEST_DIR");

fn read(rel: &str) -> String {
    fs::read_to_string(format!("{DIR}/{rel}")).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The ids of the page's controls: inputs, selects, textareas, buttons and the download link.
fn control_ids(html: &str) -> Vec<String> {
    let mut ids = vec![];
    for tag in html.split('<').skip(1) {
        let name: String = tag
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect();
        let is_control = matches!(name.as_str(), "input" | "select" | "textarea" | "button")
            || (name == "a" && tag.contains("class=\"button\""));
        let head = tag.split('>').next().unwrap_or("");
        if let (true, Some(rest)) = (is_control, head.split("id=\"").nth(1)) {
            ids.push(rest.split('"').next().unwrap().to_string());
        }
    }
    ids
}

/// `id -> (expert, beginner)` from help.js, whose entries are one line each.
fn help_entries(js: &str) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    for line in js.lines() {
        if !line.starts_with("  ") || !line.contains(": ['") {
            continue;
        }
        let quoted: Vec<&str> = line.split('\'').skip(1).step_by(2).collect();
        let (key, expert, beginner) = match quoted.as_slice() {
            // an unquoted key: `  seed: ['expert', 'beginner'],`
            [e, b] => (line.trim().split(':').next().unwrap(), *e, *b),
            // a quoted key: `  'tab-train': ['expert', 'beginner'],`
            [k, e, b] => (*k, *e, *b),
            other => panic!("help.js line has {} quoted parts: {line}", other.len()),
        };
        let previous = out.insert(key.to_string(), (expert.to_string(), beginner.to_string()));
        assert!(previous.is_none(), "help.js lists {key} twice");
    }
    out
}

fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

#[test]
fn every_control_in_the_page_has_a_help_entry() {
    let ids = control_ids(&read("web/index.html"));
    assert!(ids.len() > 25, "found only {} controls", ids.len());
    let help = help_entries(&read("web/shell/help.js"));
    let missing: Vec<&String> = ids.iter().filter(|i| !help.contains_key(*i)).collect();
    assert!(missing.is_empty(), "controls without help: {missing:?}");
}

#[test]
fn no_help_entry_describes_a_control_that_is_not_in_the_page() {
    let ids = control_ids(&read("web/index.html"));
    let help = help_entries(&read("web/shell/help.js"));
    let stale: Vec<&String> = help.keys().filter(|k| !ids.contains(k)).collect();
    assert!(
        stale.is_empty(),
        "help for controls that do not exist: {stale:?}"
    );
}

#[test]
fn each_help_text_is_a_real_explanation() {
    for (id, (expert, beginner)) in help_entries(&read("web/shell/help.js")) {
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
