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

mod every_control_in_the_page_has_a_help_entry;
