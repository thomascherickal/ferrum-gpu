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

mod the_policy_is_closed_by_default_and_allows_nothing_unsafe;
