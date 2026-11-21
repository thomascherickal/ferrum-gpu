use super::*;

#[test]
fn a_tool_that_is_not_there_stops_both_export_commands_with_its_path() {
    let d = dir("notool");
    let m = model(&d);
    let e = run(
        &w(&format!(
            "export-ollama --model {m} --name x --bin /nonexistent-ferrum/ollama"
        )),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(
        e.contains("--bin /nonexistent-ferrum/ollama: no such file"),
        "{e}"
    );
    let e = run(
        &w(&format!(
            "export-lmstudio --model {m} --bin /nonexistent-ferrum/lms"
        )),
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(
        e.contains("--bin /nonexistent-ferrum/lms: no such file"),
        "{e}"
    );
}
