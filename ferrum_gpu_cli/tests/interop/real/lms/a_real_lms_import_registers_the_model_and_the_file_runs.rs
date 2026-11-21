use super::engine::run_in_llamacpp;
use super::steps::{check_on_disk, check_registered, import};
use super::*;

#[test]
fn a_real_lms_import_registers_the_model_and_the_file_runs() {
    if !opted_in(
        "FERRUM_INTEROP_LMS",
        "imports a model into the real LM Studio library under ~/.lmstudio",
    ) {
        return;
    }
    let Some(tool) = lms() else {
        skip!("lms not installed (set FERRUM_LMS)")
    };
    let home = PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let publisher_dir = home.join(".lmstudio/models").join(PUBLISHER);
    step(format!("lms: {}", tool.display()));
    assert!(
        !publisher_dir.exists(),
        "{} already exists; remove it first (this test only deletes a folder it created)",
        publisher_dir.display()
    );
    let before = listing(&tool);
    step(format!(
        "before: `lms ls --json` lists {} models (the user's own; untouched)",
        before.len()
    ));
    assert!(before.iter().all(|(_, p)| !p.contains(PUBLISHER)));

    let f = fixture();
    let _cleanup = Cleanup(publisher_dir.clone());
    import(&f.f32, &format!("{PUBLISHER}/tiny-gpt2"), &tool);
    check_registered(&before, &listing(&tool));
    let landed = publisher_dir.join("tiny-gpt2").join("f32.gguf");
    check_on_disk(&landed, &f.f32);
    match llamacpp_dir() {
        Some(dir) => run_in_llamacpp(&dir, &landed, &ours(&f.f32)),
        None => step("llama.cpp not installed: the file is registered, the engine was not run"),
    }
}
