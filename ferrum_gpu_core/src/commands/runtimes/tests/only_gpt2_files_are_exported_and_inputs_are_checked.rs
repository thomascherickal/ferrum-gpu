use super::*;

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
    // a quote in the path cannot be put in a Modelfile (checked on the text: a quote is not a
    // legal file name on Windows)
    assert!(quotable(Path::new("/m/q\"uote.gguf"))
        .unwrap_err()
        .contains("double quote"));
    assert!(quotable(Path::new("/m/new\nline.gguf")).is_err());
    assert!(quotable(Path::new("/m/fine.gguf")).is_ok());
}
