#![forbid(unsafe_code)]
//! Guards CLAUDE.md's independence rule and the no-unsafe rule.
use std::fs;
use std::path::Path;

mod manifest;
mod source;
mod walk;
use manifest::*;
use source::*;
use walk::*;

fn scan(ext: &str, name: Option<&str>, f: &dyn Fn(&str, &Path) -> Vec<String>) -> Vec<String> {
    let mut bad = vec![];
    for p in files(ext, name) {
        if p.components().any(|c| c.as_os_str() == "test_independence") {
            continue;
        }
        for v in f(&fs::read_to_string(&p).unwrap(), p.parent().unwrap()) {
            bad.push(format!("{}: {v}", p.display()));
        }
    }
    bad
}

mod no_manifest_points_at_ferrum_cpu_or_outside;
