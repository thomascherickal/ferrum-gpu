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

mod tabs_and_panels_point_at_each_other;
mod the_page_declares_its_language_and_has_a_title;
