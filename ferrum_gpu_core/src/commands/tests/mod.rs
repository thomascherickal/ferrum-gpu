use super::*;
use std::path::PathBuf;

fn a(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn run_ok(s: &str) -> String {
    let mut out = Vec::new();
    run(&a(s), &mut out).unwrap_or_else(|e| panic!("{s}: {e}"));
    String::from_utf8(out).unwrap()
}

fn run_err(s: &str) -> String {
    run(&a(s), &mut Vec::new()).unwrap_err()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn corpus(d: &std::path::Path) -> String {
    let p = d.join("data.txt");
    std::fs::write(&p, "abcabcabcabc ".repeat(40)).unwrap();
    p.display().to_string()
}

mod budget;
mod decoder_plan;
mod files_and_outputs_that_fail_are_named;
mod fixtures;
mod gpu_weights_names_parse_and_unknown_ones_list_the_choices;
mod help_and_unknown_commands;
mod passes;
mod the_export_tools_are_looked_for_and_their_absence_is_reported;
