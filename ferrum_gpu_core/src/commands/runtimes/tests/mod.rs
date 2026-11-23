use super::*;
use crate::commands::run;
use std::ffi::OsString;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_rt_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn w(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

/// A tiny GPT-2 model file written by `train`.
fn model(d: &Path) -> String {
    let data = d.join("c.txt");
    std::fs::write(&data, "the cat sat on the mat. ".repeat(40)).unwrap();
    let m = d.join("m.gguf");
    run(
        &w(&format!("train --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 --epochs 2 --data {} --out {}", data.display(), m.display())),
        &mut Vec::new(),
    )
    .unwrap();
    m.display().to_string()
}

#[cfg(unix)]
fn script(path: &Path, body: &str) -> String {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path.display().to_string()
}

mod a_tool_that_is_not_there_stops_both_export_commands_with_its_path;
mod export_lmstudio_imports_then_waits_for_the_listing;
mod only_gpt2_files_are_exported_and_inputs_are_checked;
mod the_modelfile_matches_ferrum_cpus_text;
mod tools_are_found_in_flag_env_path_then_home_order;
mod unusable_inputs_and_unwritable_outputs_are_named;
