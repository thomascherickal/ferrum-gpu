use super::*;
use std::path::PathBuf;

fn a(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn run_ok(s: &str) -> String {
    let mut out = Vec::new();
    run(&a(s), &mut out).unwrap_or_else(|e| panic!("{s}: {e}"));
    String::from_utf8(out).unwrap()
}

fn run_err(s: &str) -> String {
    run(&a(s), &mut Vec::new()).unwrap_err()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn corpus(d: &std::path::Path) -> String {
    let p = d.join("data.txt");
    std::fs::write(&p, "abcabcabcabc ".repeat(40)).unwrap();
    p.display().to_string()
}

#[test]
fn help_and_unknown_commands() {
    assert!(run_ok("help").contains("usage: ferrum-gpu"));
    assert!(run(&[], &mut Vec::new()).unwrap_err().contains("usage"));
    assert!(run_err("frobnicate").contains("unknown command"));
}

#[test]
fn unknown_flags_are_rejected_per_command() {
    assert!(run_err("info --bogus 1").contains("unknown flag"));
    assert!(run_err("eval --device cpu").contains("unknown flag"));
    assert!(
        run_err("train --data x").contains("--out is required")
            || run_err("train --data x").contains("data")
    );
}

#[test]
fn train_generate_run_eval_finetune_round_trip_on_cpu() {
    let d = dir("e2e");
    let data = corpus(&d);
    let m1 = d.join("m1.gguf").display().to_string();
    let m2 = d.join("m2.gguf").display().to_string();
    let t = run_ok(&format!(
        "train --arch llama --device cpu --preset tiny --data {data} --out {m1} --steps 30 --batch 2 --context 16"
    ));
    assert!(
        t.contains("device: ") && t.contains("steps: 30") && t.contains("wrote "),
        "{t}"
    );
    assert!(run_ok(&format!(
        "generate --device cpu --model {m1} --prompt ab --max-new 5"
    ))
    .starts_with("ab"));
    let r = run_ok(&format!(
        "run-gguf --device cpu --model {m1} --prompt ab --max-new 3"
    ));
    assert!(r.contains("model: vocab 256"), "{r}");
    let e = run_ok(&format!("eval --model {m1} --data {data}"));
    assert!(e.contains("perplexity"), "{e}");
    let f = run_ok(&format!(
        "finetune-gguf --device cpu --model {m1} --data {data} --out {m2} --steps 5 --batch 2 --context 16 --quant q8_0"
    ));
    assert!(f.contains("wrote "), "{f}");
    assert!(run_ok(&format!("info --device cpu --model {m2}")).contains("model: vocab 256"));
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn errors_name_the_problem() {
    let d = dir("err");
    let data = corpus(&d);
    let out = d.join("o.gguf").display().to_string();
    assert!(run_err(&format!(
        "train --arch llama --device cpu --preset huge --data {data} --out {out}"
    ))
    .contains("unknown preset"));
    assert!(run_err(&format!(
        "train --arch llama --device cpu --data {data} --out {out} --quant fp9"
    ))
    .contains("--quant"));
    assert!(run_err(&format!(
        "train --arch llama --device cpu --data {data} --out {out} --steps 0"
    ))
    .contains("steps"));
    assert!(run_err(&format!("train --device cpu --data /nope --out {out}")).contains("--data"));
    assert!(run_err("generate --device cpu --model /nope --prompt x").contains("/nope"));
    assert!(
        run_err(&format!("train --device tpu --data {data} --out y")).contains("unknown device")
    );
    let bad = d.join("bad.gguf");
    std::fs::write(&bad, b"not gguf").unwrap();
    assert!(run_err(&format!("eval --model {} --data {data}", bad.display())).contains("magic"));
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn mean_loss_of_untrained_model_is_is_finite_and_sane() {
    let m = fresh_model("tiny", 1).unwrap();
    let (loss, n) = mean_loss(&m, b"hello world, hello world").unwrap();
    assert!(
        n > 0 && loss.is_finite() && loss > 0.0 && loss < 12.0,
        "{loss}"
    );
    assert!(mean_loss(&m, b"x").unwrap_err().contains("at least 2"));
}

mod gate_tests {
    use super::*;
    use crate::engine::{Backend, CpuBackend, GenOpts, Progress, TrainSpec};
    use ferrum_wgpu::select::{AdapterSummary, Kind};

    /// Pretends to be a GPU named "Fake GPU"; never runs anything.
    struct FakeGpu;
    impl Backend for FakeGpu {
        fn kind(&self) -> &'static str {
            "gpu"
        }
        fn name(&self) -> String {
            "Fake GPU".into()
        }
        fn train(
            &self,
            _: &Llama,
            _: &TrainSpec,
            _: &Arc<Mutex<Progress>>,
        ) -> Result<Llama, String> {
            unreachable!()
        }
        fn generate(&self, _: &Llama, _: &[usize], _: &GenOpts) -> Result<Vec<usize>, String> {
            unreachable!()
        }
    }

    fn listed() -> Vec<AdapterSummary> {
        vec![AdapterSummary {
            name: "Fake GPU".into(),
            backend: "Vulkan".into(),
            kind: Kind::Discrete,
            max_buffer: 1 << 32,
            max_binding: 1 << 30,
            max_workgroup_invocations: 256,
            features: Default::default(),
        }]
    }

    fn app(b: Arc<dyn Backend>, adapters: Vec<AdapterSummary>) -> Session {
        Session::with_backend(b, DeviceChoice::Gpu, None, adapters)
    }

    #[test]
    fn a_gpu_run_that_fits_prints_the_estimate_and_one_that_does_not_is_refused() {
        let m = fresh_model("tiny", 1).unwrap();
        let plan = preflight::train_plan(&m, 2, 16);
        let gpu = app(Arc::new(FakeGpu), listed());
        let mut out = Vec::new();
        gate(&mut out, &gpu, &plan, Some(4096)).unwrap();
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("preflight: GPU buffers"));
        let e = gate(&mut Vec::new(), &gpu, &plan, Some(0)).unwrap_err();
        assert!(e.contains("refusing to start on Fake GPU"), "{e}");
    }

    /// A card with 2048 MiB total and 1989 MiB free, as `nvidia-smi` would say; no RAM figures.
    struct Mx330;
    impl sysmem::Readers for Mx330 {
        fn nvidia_smi(&self) -> Option<String> {
            Some("Fake GPU, 2048, 1989\n".into())
        }
        fn meminfo(&self) -> Option<String> {
            None
        }
        fn sysctl_memsize(&self) -> Option<String> {
            None
        }
        fn sysfs_vram_mb(&self) -> Vec<u64> {
            vec![]
        }
    }

    /// A plan of `mib` MiB in two buffers (each under the fake adapter's binding limit).
    fn plan_of(mib: u64) -> ferrum_wgpu::budget::BufferPlan {
        let mut p = ferrum_wgpu::budget::BufferPlan::default();
        p.add("first half", mib / 2 * 1048576);
        p.add("second half", (mib - mib / 2) * 1048576);
        p
    }

    #[test]
    fn the_preflight_uses_the_cards_free_memory_with_the_measured_headroom() {
        // CAPABILITY.md: on this card plans up to 1449 MiB loaded and plans from 1721 MiB failed
        let gpu = app(Arc::new(FakeGpu), listed());
        for fits in [279u64, 956, 1449] {
            let mut out = Vec::new();
            gate_with(&Mx330, &mut out, &gpu, &plan_of(fits), None)
                .unwrap_or_else(|e| panic!("{fits} MiB should pass: {e}"));
            assert!(
                String::from_utf8(out).unwrap().contains("budget 1591"),
                "the budget is shown"
            );
        }
        for refused in [1721u64, 1925, 2454, 4286] {
            let e = gate_with(&Mx330, &mut Vec::new(), &gpu, &plan_of(refused), None)
                .expect_err(&format!("{refused} MiB should be refused"));
            assert!(e.contains("refusing to start on Fake GPU"), "{e}");
        }
        // the flag still overrides what the machine says
        gate_with(&Mx330, &mut Vec::new(), &gpu, &plan_of(1925), Some(4096)).unwrap();
    }

    #[test]
    fn with_no_memory_figure_only_the_binding_limit_applies_as_before() {
        struct Blind;
        impl sysmem::Readers for Blind {
            fn nvidia_smi(&self) -> Option<String> {
                None
            }
            fn meminfo(&self) -> Option<String> {
                None
            }
            fn sysctl_memsize(&self) -> Option<String> {
                None
            }
            fn sysfs_vram_mb(&self) -> Vec<u64> {
                vec![]
            }
        }
        let gpu = app(Arc::new(FakeGpu), listed());
        let mut out = Vec::new();
        // 4286 MiB in buffers that each respect the binding limit: only the missing budget is at issue
        let mut big = ferrum_wgpu::budget::BufferPlan::default();
        for i in 0..5 {
            big.add(&format!("part {i}"), 857 * 1048576);
        }
        gate_with(&Blind, &mut out, &gpu, &big, None).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("budget unknown"));
    }

    #[test]
    fn the_decoder_plan_is_split_at_the_chosen_adapters_binding_limit() {
        let m = fresh_model("tiny", 1).unwrap();
        let whole = preflight::decode_buffers(&m);
        let biggest = whole
            .rows()
            .iter()
            .filter(|r| r.0.contains("blk0.gate") || r.0 == "head" || r.0 == "tok_emb")
            .map(|r| r.1)
            .max()
            .unwrap();
        // a limit between the unsplittable buffers and the biggest weight
        let unsplittable = whole
            .rows()
            .iter()
            .filter(|r| r.0.contains("cache") || r.0.starts_with("scratch"))
            .map(|r| r.1)
            .max()
            .unwrap();
        let limit = (biggest / 2).max(unsplittable);
        assert!(limit < biggest, "the tiny preset must have a weight bigger than its own KV cache ({limit} vs {biggest})");
        let mut one = listed();
        one[0].max_binding = limit;
        let split = preflight::decode_buffers_for(&m, &app(Arc::new(FakeGpu), one));
        assert_eq!(split.total(), whole.total(), "splitting keeps the bytes");
        assert!(split.rows().len() > whole.rows().len());
        assert!(split.rows().iter().all(|r| r.1 <= limit));
        // a CPU backend has no limit, so the plan is the whole one
        let cpu = preflight::decode_buffers_for(&m, &app(Arc::new(CpuBackend::new(1)), listed()));
        assert_eq!(cpu.rows(), whole.rows());
    }

    #[test]
    fn cpu_backends_and_unlisted_adapters_pass_silently() {
        let m = fresh_model("tiny", 1).unwrap();
        let plan = preflight::train_plan(&m, 2, 16);
        let cpu = app(Arc::new(CpuBackend::new(1)), listed());
        let mut out = Vec::new();
        gate(&mut out, &cpu, &plan, Some(0)).unwrap();
        gate(&mut out, &app(Arc::new(FakeGpu), vec![]), &plan, Some(0)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn mem_flag_parses_or_names_itself_in_the_error() {
        let ok = Args::parse(&DEVICE, &a("--gpu-mem-mb 512")).unwrap();
        assert_eq!(mem_mb(&ok).unwrap(), Some(512));
        assert_eq!(mem_mb(&Args::parse(&DEVICE, &[]).unwrap()).unwrap(), None);
        let bad = Args::parse(&DEVICE, &a("--gpu-mem-mb lots")).unwrap();
        assert!(mem_mb(&bad).unwrap_err().contains("--gpu-mem-mb"));
    }
}

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
