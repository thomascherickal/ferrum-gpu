use super::*;
use crate::commands::runtimes::export_ollama::write_modelfile;
use ferrum_gguf::{GgufBuilder, Value};

#[test]
fn a_model_that_is_not_gguf_or_lacks_its_context_length_is_named() {
    let d = dir("unusable");
    let text = d.join("not.gguf");
    std::fs::write(&text, "plain text").unwrap();
    let e = run(
        &w(&format!(
            "export-ollama --model {} --name x",
            text.display()
        )),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(
        e.starts_with(&format!("cannot read GGUF {}: ", text.display())),
        "{e}"
    );
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    let bare = d.join("bare.gguf");
    std::fs::write(&bare, b.into_bytes()).unwrap();
    let e = run(
        &w(&format!(
            "export-ollama --model {} --name x",
            bare.display()
        )),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(
        e.starts_with(&format!("{}: ", bare.display())) && e.contains("context_length"),
        "{e}"
    );
}

#[test]
fn a_modelfile_that_cannot_be_written_is_named() {
    let d = dir("unwritable");
    let m = model(&d);
    let out = "/nonexistent-ferrum-dir/Modelfile";
    let e = run(
        &w(&format!(
            "export-ollama --model {m} --name x --modelfile-only {out}"
        )),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(e.starts_with(&format!("{out}: ")), "{e}");
    // the scratch directory is a regular file; then the Modelfile's own path is a directory
    let file = d.join("a_file");
    std::fs::write(&file, "x").unwrap();
    let e = write_modelfile(&file, "n", "text").unwrap_err();
    assert!(e.starts_with(&format!("{}: ", file.display())), "{e}");
    std::fs::create_dir_all(d.join("n.Modelfile")).unwrap();
    let e = write_modelfile(&d, "n", "text").unwrap_err();
    assert!(
        e.starts_with(&format!("{}: ", d.join("n.Modelfile").display())),
        "{e}"
    );
    assert!(write_modelfile(&d, "other", "text").unwrap().is_file());
}
