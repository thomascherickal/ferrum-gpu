//! End-to-end: the real `ferrum-gpu` binary on a tiny corpus. The GPU leg skips (printing why)
//! when no hardware adapter exists.

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrum-gpu"))
        .args(args)
        .output()
        .expect("run ferrum-gpu")
}

fn ok(args: &[&str]) -> String {
    let o = bin(args);
    assert!(
        o.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap()
}

fn work(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_e2e_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn loss_pair(report: &str) -> (f32, f32) {
    let l = report
        .lines()
        .find(|l| l.starts_with("steps:"))
        .expect("steps line");
    let v: Vec<f32> = l
        .split_whitespace()
        .filter_map(|w| w.parse().ok())
        .collect();
    (v[v.len() - 2], v[v.len() - 1])
}

#[test]
fn train_eval_generate_finetune_and_refusals_on_the_cpu() {
    let d = work("cpu");
    let data = d.join("data.txt");
    std::fs::write(&data, "the cat sat on the mat. ".repeat(60)).unwrap();
    let (data, m1, m2) = (data.to_str().unwrap(), d.join("m1.gguf"), d.join("m2.gguf"));
    let (m1, m2) = (m1.to_str().unwrap(), m2.to_str().unwrap());

    let t = ok(&[
        "train",
        "--device",
        "cpu",
        "--preset",
        "tiny",
        "--data",
        data,
        "--out",
        m1,
        "--steps",
        "60",
        "--batch",
        "4",
        "--context",
        "32",
    ]);
    let (first, last) = loss_pair(&t);
    assert!(last < first, "loss should fall: {t}");
    let e = ok(&["eval", "--model", m1, "--data", data]);
    assert!(e.contains("perplexity"), "{e}");
    let g = ok(&[
        "generate",
        "--device",
        "cpu",
        "--model",
        m1,
        "--prompt",
        "the c",
        "--max-new",
        "8",
    ]);
    assert!(g.starts_with("the c"), "{g}");
    assert_eq!(
        g,
        ok(&[
            "generate",
            "--device",
            "cpu",
            "--model",
            m1,
            "--prompt",
            "the c",
            "--max-new",
            "8"
        ]),
        "greedy is deterministic"
    );
    let f = ok(&[
        "finetune-gguf",
        "--device",
        "cpu",
        "--model",
        m1,
        "--data",
        data,
        "--out",
        m2,
        "--steps",
        "10",
        "--batch",
        "4",
        "--context",
        "32",
        "--quant",
        "q8_0",
    ]);
    assert!(f.contains("wrote"), "{f}");
    assert!(ok(&["info", "--device", "cpu", "--model", m2]).contains("model: vocab 256"));

    // refusals exit non-zero with a message on stderr
    for bad in [
        vec!["train", "--bogus"],
        vec!["nonsense"],
        vec![
            "generate", "--device", "cpu", "--model", "/nope", "--prompt", "x",
        ],
    ] {
        let o = bin(&bad);
        assert!(!o.status.success() && !o.stderr.is_empty(), "{bad:?}");
    }
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn gpu_train_and_preflight_refusal_when_a_hardware_adapter_exists() {
    let info = ok(&["info", "--device", "cpu"]);
    if !info
        .lines()
        .any(|l| l.starts_with("adapter:") && (l.contains("Discrete") || l.contains("Integrated")))
    {
        eprintln!("skipping: no hardware GPU adapter reported by `info`:\n{info}");
        return;
    }
    let d = work("gpu");
    let data = d.join("data.txt");
    std::fs::write(&data, "abcabcabc ".repeat(80)).unwrap();
    let (data, out) = (data.to_str().unwrap(), d.join("g.gguf"));
    let out = out.to_str().unwrap();
    let t = ok(&[
        "train",
        "--device",
        "gpu",
        "--preset",
        "tiny",
        "--data",
        data,
        "--out",
        out,
        "--steps",
        "20",
        "--batch",
        "2",
        "--context",
        "16",
        "--gpu-mem-mb",
        "1024",
    ]);
    assert!(
        t.contains("(gpu)") && t.contains("preflight: GPU buffers"),
        "{t}"
    );
    let o = bin(&[
        "train",
        "--device",
        "gpu",
        "--preset",
        "tiny",
        "--data",
        data,
        "--out",
        out,
        "--steps",
        "2",
        "--gpu-mem-mb",
        "0",
    ]);
    assert!(!o.status.success());
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("refusing to start"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    std::fs::remove_dir_all(d).ok();
}
