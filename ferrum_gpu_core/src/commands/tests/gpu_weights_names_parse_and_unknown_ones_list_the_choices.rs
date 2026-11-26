use super::*;

#[test]
fn gpu_weights_names_parse_and_unknown_ones_list_the_choices() {
    use ferrum_wgpu::ops_quant::QKind;
    assert_eq!(gpu_weights(None).unwrap(), None);
    assert_eq!(gpu_weights(Some("f32")).unwrap(), None);
    assert_eq!(gpu_weights(Some("f16")).unwrap(), Some(QKind::F16));
    assert_eq!(gpu_weights(Some("q8_0")).unwrap(), Some(QKind::Q8_0));
    assert_eq!(gpu_weights(Some("q4_0")).unwrap(), Some(QKind::Q4_0));
    for (name, kind) in [
        ("fp4", QKind::Fp4),
        ("mxfp4", QKind::Fp4),
        ("fp8", QKind::Fp8E4M3),
        ("fp8e4m3", QKind::Fp8E4M3),
        ("fp8e5m2", QKind::Fp8E5M2),
        ("q2", QKind::Q2K),
        ("q2_k", QKind::Q2K),
    ] {
        assert_eq!(gpu_weights(Some(name)).unwrap(), Some(kind), "{name}");
    }
    let e = gpu_weights(Some("int3")).unwrap_err();
    assert!(
        e.contains("--gpu-weights") && e.contains("f32, f16, q8_0, q4_0, fp4"),
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
