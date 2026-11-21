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

mod gpu_train_and_preflight_refusal_when_a_hardware_adapter_exists;
mod train_eval_generate_finetune_and_refusals_on_the_cpu;
