//! The page runs under a strict Content-Security-Policy (the desktop app loads it in a window), so
//! the policy must be on every static response and the page and its scripts must not need anything
//! the policy forbids: no inline script or style, no event-handler attributes, no `eval`.

use ferrum_gpu_gui::assets::{serve, CSP, MODULES};
use std::fs;

const DIR: &str = env!("CARGO_MANIFEST_DIR");

fn read(rel: &str) -> String {
    fs::read_to_string(format!("{DIR}/{rel}")).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn header<'a>(r: &'a ferrum_gpu_gui::http::Response, name: &str) -> Option<&'a str> {
    r.headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

#[test]
fn the_policy_is_closed_by_default_and_allows_nothing_unsafe() {
    for d in [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ] {
        assert!(CSP.contains(d), "missing {d}");
    }
    for bad in ["unsafe-inline", "unsafe-eval", "http:", "https:", "*"] {
        assert!(!CSP.contains(bad), "the policy must not contain {bad}");
    }
}

#[test]
fn every_static_response_carries_the_policy() {
    let mut paths: Vec<String> = [
        "/",
        "/index.html",
        "/style.css",
        "/app.js",
        "/shaders/prelude.wgsl",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    paths.extend(MODULES.iter().map(|(n, _)| format!("/web/{n}")));
    for p in paths {
        let r = serve(&p).unwrap_or_else(|| panic!("{p} is not served"));
        assert_eq!(header(&r, "Content-Security-Policy"), Some(CSP), "{p}");
        // nosniff is added to every response by the HTTP layer itself (see ferrum_gpu_core::http)
    }
}

#[test]
fn the_page_has_no_inline_script_style_or_handler() {
    let html = read("web/index.html");
    assert_eq!(html.matches("<script").count(), 1, "one script tag");
    assert!(html.contains(r#"<script type="module" src="/app.js">"#));
    assert!(!html.contains(" style="), "no inline style attributes");
    assert!(!html.contains("<style"), "no style element");
    for ev in [
        "onclick=",
        "onload=",
        "onchange=",
        "oninput=",
        "onsubmit=",
        "onerror=",
        "javascript:",
    ] {
        assert!(!html.contains(ev), "the page must not use {ev}");
    }
}

#[test]
fn no_script_needs_eval_or_a_dynamic_function() {
    let mut files = vec!["web/app.js".to_string()];
    files.extend(MODULES.iter().map(|(n, _)| format!("web/{n}")));
    for f in files {
        let js = read(&f);
        for bad in [
            "eval(",
            "new Function(",
            "setTimeout(\"",
            "setInterval(\"",
            "document.write(",
        ] {
            assert!(!js.contains(bad), "{f} uses {bad}");
        }
    }
}
