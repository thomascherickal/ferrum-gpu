use super::*;

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
