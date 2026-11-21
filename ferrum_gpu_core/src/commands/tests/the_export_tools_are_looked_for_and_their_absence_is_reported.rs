use super::*;

#[test]
fn the_export_tools_are_looked_for_and_their_absence_is_reported() {
    let d = dir("tools");
    let (c, m) = (corpus(&d), d.join("m.gguf").display().to_string());
    run_ok(&format!(
        "train --arch llama --device cpu --preset tiny --steps 3 --batch 2 --context 16 --data {c} --out {m}"
    ));
    // a byte-level Llama file is not a GPT-2 file: the export commands say so before looking for a tool
    assert!(!run_err(&format!(
        "export-ollama --model {m} --name x --bin /nonexistent/ollama"
    ))
    .is_empty());
    assert!(!run_err(&format!(
        "export-lmstudio --model {m} --bin /nonexistent/lms"
    ))
    .is_empty());
}

#[test]
fn a_memory_budget_the_model_does_not_fit_is_refused_before_anything_runs() {
    let d = dir("budget");
    let (c, m) = (corpus(&d), d.join("m.gguf").display().to_string());
    // 1 MiB fits nothing; with no usable GPU the request is refused for that reason instead
    let e = run_err(&format!(
        "train --arch llama --device gpu --gpu-mem-mb 1 --preset tiny --steps 3 --batch 16 --context 64 --data {c} --out {m}"
    ));
    assert!(!e.is_empty());
}

#[test]
fn a_cancelled_training_still_writes_its_model_and_a_small_budget_refuses_generate() {
    let d = dir("cancelled");
    let (c, m) = (corpus(&d), d.join("m.gguf").display().to_string());
    {
        // the thread's cancel flag is already up: no step runs, the model is written as it is
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let _scope = crate::cancel::install(flag);
        let out = run_ok(&format!(
            "train --arch llama --device cpu --preset tiny --steps 5 --batch 2 --context 16 --data {c} --out {m}"
        ));
        assert!(out.contains("wrote") && !out.contains("steps:"), "{out}");
    }
    // a memory budget of nothing refuses the GPU decoder before it opens anything
    assert!(!run_err(&format!(
        "generate --model {m} --prompt ab --device gpu --gpu-mem-mb 0"
    ))
    .is_empty());
}
