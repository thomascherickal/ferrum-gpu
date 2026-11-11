use super::*;
use crate::cmds::run;
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

#[test]
fn the_modelfile_matches_ferrum_cpus_text() {
    assert_eq!(
        modelfile(Path::new("/m/a b.gguf"), 64),
        "FROM \"/m/a b.gguf\"\nTEMPLATE \"{{ .Prompt }}\"\nPARAMETER stop \"<|endoftext|>\"\nPARAMETER temperature 0.8\nPARAMETER num_ctx 64\n"
    );
}

#[test]
fn modelfile_only_writes_the_file_without_any_tool() {
    let d = dir("mfonly");
    let m = model(&d);
    let out = d.join("m.Modelfile");
    let mut o = Vec::new();
    run(
        &w(&format!(
            "export-ollama --model {m} --modelfile-only {}",
            out.display()
        )),
        &mut o,
    )
    .unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.contains("num_ctx 16") && text.starts_with("FROM \""),
        "{text}"
    );
    assert!(String::from_utf8(o).unwrap().starts_with("wrote "));
}

#[cfg(unix)]
#[test]
fn export_ollama_runs_create_with_the_modelfile_and_reports_failures() {
    let d = dir("ollama");
    let m = model(&d);
    let log = d.join("args.txt");
    let good = script(
        &d.join("good"),
        &format!("echo \"$@\" > {0}; cat \"$4\" >> {0}", log.display()),
    );
    let mut o = Vec::new();
    run(
        &w(&format!(
            "export-ollama --model {m} --name mytiny --bin {good}"
        )),
        &mut o,
    )
    .unwrap();
    assert!(String::from_utf8(o)
        .unwrap()
        .contains("created Ollama model 'mytiny'"));
    let got = std::fs::read_to_string(&log).unwrap();
    assert!(
        got.starts_with("create mytiny -f ") && got.contains("PARAMETER num_ctx 16"),
        "{got}"
    );
    // a failing tool; and the hint when the server is not running
    let bad = script(
        &d.join("bad"),
        "echo 'Error: could not connect to ollama' >&2; exit 1",
    );
    let e = run(
        &w(&format!("export-ollama --model {m} --name x --bin {bad}")),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(
        e.contains("`ollama create x` failed") && e.contains("ollama serve"),
        "{e}"
    );
    let other = script(&d.join("other"), "echo nope >&2; exit 2");
    let e = run(
        &w(&format!("export-ollama --model {m} --name x --bin {other}")),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(e.contains("nope") && !e.contains("ollama serve"), "{e}");
}

#[test]
fn only_gpt2_files_are_exported_and_inputs_are_checked() {
    let d = dir("refuse");
    let l = d.join("l.gguf");
    let data = d.join("c.txt");
    std::fs::write(&data, "abcabcabc ".repeat(40)).unwrap();
    run(&w(&format!("train --arch llama --device cpu --preset tiny --data {} --out {} --steps 2 --batch 2 --context 8", data.display(), l.display())), &mut Vec::new()).unwrap();
    let e = run(
        &w(&format!(
            "export-ollama --model {} --name x --modelfile-only {}",
            l.display(),
            d.join("o").display()
        )),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(e.contains("architecture 'llama' cannot be exported"), "{e}");
    assert!(
        run(&w("export-ollama --model /nope --name x"), &mut Vec::new())
            .unwrap_err()
            .contains("cannot read GGUF /nope")
    );
    assert!(run(
        &w(&format!("export-lmstudio --model {}", l.display())),
        &mut Vec::new()
    )
    .unwrap_err()
    .contains("cannot be exported"));
    let m = model(&d);
    assert!(
        run(&w(&format!("export-ollama --model {m}")), &mut Vec::new())
            .unwrap_err()
            .contains("--name is required")
    );
    assert!(run(
        &w(&format!("export-lmstudio --model {m} --repo nopublisher")),
        &mut Vec::new()
    )
    .unwrap_err()
    .contains("<publisher>/<name>"));
    assert!(run(
        &w(&format!("export-lmstudio --model {m} --repo a/b/c")),
        &mut Vec::new()
    )
    .unwrap_err()
    .contains("<publisher>/<name>"));
    // a quote in the path cannot be put in a Modelfile
    let q = d.join("q\"uote.gguf");
    std::fs::copy(&m, &q).unwrap();
    let mut a = w("export-ollama --model");
    a.push(q.display().to_string());
    a.extend(w("--modelfile-only x"));
    assert!(run(&a, &mut Vec::new())
        .unwrap_err()
        .contains("double quote"));
}

#[test]
fn tools_are_found_in_flag_env_path_then_home_order() {
    let d = dir("find");
    let t = d.join("tool");
    std::fs::write(&t, "x").unwrap();
    let tool = t.to_str().unwrap();
    let path_var: OsString = d.clone().into_os_string();
    assert_eq!(
        find_tool_in(Some(tool), None, "E", None, "tool", "x").unwrap(),
        t
    );
    assert!(find_tool_in(Some("/no/such"), None, "E", None, "tool", "x")
        .unwrap_err()
        .contains("--bin /no/such"));
    assert_eq!(
        find_tool_in(None, Some(tool), "E", None, "tool", "x").unwrap(),
        t
    );
    assert!(
        find_tool_in(None, Some("/no/such"), "FERRUM_X", None, "tool", "x")
            .unwrap_err()
            .contains("FERRUM_X=/no/such")
    );
    assert_eq!(
        find_tool_in(None, Some(""), "E", Some(&path_var), "tool", "x").unwrap(),
        t,
        "an empty variable is ignored"
    );
    let e = find_tool_in(
        None,
        None,
        "FERRUM_X",
        Some(&OsString::from("/nonexistent")),
        "tool",
        "no/home/fallback",
    )
    .unwrap_err();
    assert!(
        e.contains("cannot find `tool`") && e.contains("FERRUM_X"),
        "{e}"
    );
}

#[test]
fn the_lms_listing_is_matched_by_path_suffix() {
    let json = "Waking up LM Studio service...\n[{\"path\":\"other/x.gguf\",\"modelKey\":\"no\"},{\"path\":\"/home/u/.lmstudio/models/ferrum/m/m.gguf\",\"modelKey\":\"ferrum/m\"}]";
    assert_eq!(
        model_key_for(json, "ferrum/m/m.gguf").as_deref(),
        Some("ferrum/m")
    );
    assert_eq!(
        model_key_for(json, "errum/m/m.gguf"),
        None,
        "the match is on a / boundary"
    );
    assert_eq!(
        model_key_for(
            "{\"path\":\"ferrum/m/m.gguf\",\"modelKey\":\"k\"}",
            "ferrum/m/m.gguf"
        )
        .as_deref(),
        Some("k")
    );
    assert_eq!(model_key_for("no json at all", "x"), None);
    assert_eq!(model_key_for("[]", "x"), None);
    assert_eq!(
        model_key_for("[{\"path\":\"a/b\"}]", "a/b"),
        None,
        "an entry without a modelKey"
    );
}

#[cfg(unix)]
#[test]
fn export_lmstudio_imports_then_waits_for_the_listing() {
    let d = dir("lms");
    let m = model(&d);
    let name = Path::new(&m)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let log = d.join("log.txt");
    let listing = format!("[{{\"path\":\"/x/ferrum/m/{name}\",\"modelKey\":\"ferrum/m\"}}]");
    let lms = script(
        &d.join("lms"),
        &format!(
            "echo \"$@\" >> {}\nif [ \"$1\" = ls ]; then echo '{listing}'; fi",
            log.display()
        ),
    );
    let a = Args::parse(
        &[
            crate::flags::val("--model"),
            crate::flags::val("--repo"),
            crate::flags::val("--bin"),
        ],
        &w(&format!("--model {m} --repo ferrum/m --bin {lms}")),
    )
    .unwrap();
    let mut o = Vec::new();
    export_lmstudio_with(&a, &mut o, 3, std::time::Duration::from_millis(1)).unwrap();
    assert!(String::from_utf8(o).unwrap().contains("'ferrum/m'"));
    let calls = std::fs::read_to_string(&log).unwrap();
    assert!(
        calls.starts_with("import -y -l --user-repo ferrum/m ") && calls.contains("\nls --json"),
        "{calls}"
    );
    // an import that fails, and a listing that never shows the model
    let fail = script(&d.join("fail"), "echo broken >&2; exit 1");
    let af = Args::parse(
        &[crate::flags::val("--model"), crate::flags::val("--bin")],
        &w(&format!("--model {m} --bin {fail}")),
    )
    .unwrap();
    assert!(
        export_lmstudio_with(&af, &mut Vec::new(), 1, std::time::Duration::ZERO)
            .unwrap_err()
            .contains("`lms import` failed")
    );
    let empty = script(&d.join("empty"), "if [ \"$1\" = ls ]; then echo '[]'; fi");
    let ae = Args::parse(
        &[crate::flags::val("--model"), crate::flags::val("--bin")],
        &w(&format!("--model {m} --bin {empty}")),
    )
    .unwrap();
    assert!(
        export_lmstudio_with(&ae, &mut Vec::new(), 2, std::time::Duration::from_millis(1))
            .unwrap_err()
            .contains("never listed it")
    );
    // without --repo the repo is ferrum/<file stem>
    let ad = Args::parse(
        &[crate::flags::val("--model"), crate::flags::val("--bin")],
        &w(&format!("--model {m} --bin {lms}")),
    )
    .unwrap();
    export_lmstudio_with(&ad, &mut Vec::new(), 3, std::time::Duration::from_millis(1)).unwrap();
    assert!(std::fs::read_to_string(&log)
        .unwrap()
        .contains("--user-repo ferrum/m "));
}
