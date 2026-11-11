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
        "train --device cpu --preset tiny --data {data} --out {m1} --steps 30 --batch 2 --context 16"
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
        "train --device cpu --preset huge --data {data} --out {out}"
    ))
    .contains("unknown preset"));
    assert!(run_err(&format!(
        "train --device cpu --data {data} --out {out} --quant q2"
    ))
    .contains("--quant"));
    assert!(run_err(&format!(
        "train --device cpu --data {data} --out {out} --steps 0"
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
    use ferrum_gpu_gui::engine::{Backend, CpuBackend, GenOpts, Progress, TrainSpec};
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
        }]
    }

    fn app(b: Arc<dyn Backend>, adapters: Vec<AdapterSummary>) -> App {
        App::with_backend(b, DeviceChoice::Gpu, None, adapters)
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
