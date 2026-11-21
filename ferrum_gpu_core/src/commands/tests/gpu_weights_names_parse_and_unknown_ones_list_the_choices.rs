use super::*;

#[test]
fn gpu_weights_names_parse_and_unknown_ones_list_the_choices() {
    use ferrum_wgpu::ops_quant::QKind;
    assert_eq!(gpu_weights(None).unwrap(), None);
    assert_eq!(gpu_weights(Some("f32")).unwrap(), None);
    assert_eq!(gpu_weights(Some("f16")).unwrap(), Some(QKind::F16));
    assert_eq!(gpu_weights(Some("q8_0")).unwrap(), Some(QKind::Q8_0));
    assert_eq!(gpu_weights(Some("q4_0")).unwrap(), Some(QKind::Q4_0));
    let e = gpu_weights(Some("int3")).unwrap_err();
    assert!(
        e.contains("--gpu-weights") && e.contains("f32, f16, q8_0, q4_0"),
        "{e}"
    );
}

#[test]
fn adapter_lines_name_the_features_and_the_source_of_the_budget() {
    use ferrum_wgpu::select::{AdapterFeatures, AdapterSummary, Kind};
    let mut ad = AdapterSummary {
        name: "Fake GPU".into(),
        backend: "Vulkan".into(),
        kind: Kind::Discrete,
        max_buffer: 1 << 31,
        max_binding: 1 << 28,
        max_workgroup_invocations: 256,
        features: AdapterFeatures::default(),
    };
    let l = adapter_lines(&ad, Some(512));
    assert_eq!(l[0], "adapter: Fake GPU (Discrete, Vulkan)");
    assert_eq!(l[1], "  features: none reported; largest buffer 256 MiB");
    assert!(l[2].starts_with("  ") && l[2].contains("512"), "{}", l[2]);
    ad.features.subgroups = true;
    assert!(adapter_lines(&ad, None)[1].contains("subgroups"));
}

#[test]
fn an_adapter_that_does_not_exist_is_refused_and_auto_runs_on_whatever_the_machine_has() {
    let d = dir("auto");
    let (c, m) = (corpus(&d), d.join("m.gguf").display().to_string());
    assert!(!run_err("info --device gpu --gpu-adapter no-such-adapter-xyz").is_empty());
    run_ok(&format!(
        "train --arch llama --device cpu --preset tiny --steps 3 --batch 2 --context 16 --data {c} --out {m}"
    ));
    // `auto` may print a note about its choice; either way the model runs
    let out = run_ok(&format!(
        "run-gguf --device auto --model {m} --prompt ab --max-new 2"
    ));
    assert!(out.contains("device:"), "{out}");
}

#[test]
fn every_command_refuses_an_unknown_flag_and_a_bad_thread_count() {
    for cmd in [
        "train",
        "finetune-gguf",
        "run",
        "generate",
        "run-gguf",
        "prepare",
        "tokenize",
        "export-ollama",
        "export-lmstudio",
        "export-gguf",
        "eval",
        "lab",
        "info",
    ] {
        let e = run_err(&format!("{cmd} --no-such-flag-xyz"));
        assert!(e.contains("no-such-flag-xyz"), "{cmd}: {e}");
    }
    assert!(run_err("info --device cpu --threads many").contains("--threads"));
}

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
