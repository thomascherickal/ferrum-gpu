//! `train --tie` and `--dropout` on a GPU adapter, through the real command. Uses llvmpipe (the
//! software Vulkan adapter) and skips, printing why, where it does not exist.

use crate::cmds::run;
use std::path::Path;

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn have_llvmpipe() -> bool {
    ferrum_wgpu::Gpu::list()
        .iter()
        .any(|a| a.name.to_lowercase().contains("llvmpipe"))
}

fn train(device: &str, extra: &str, data: &Path, out: &Path) -> Result<String, String> {
    let mut buf = Vec::new();
    run(
        &words(&format!(
            "train {device} --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 \
             --epochs 3 --vocab 300 --seed 5 {extra} --data {} --out {}",
            data.display(),
            out.display()
        )),
        &mut buf,
    )?;
    Ok(String::from_utf8(buf).unwrap())
}

fn has_own_head(p: &Path) -> bool {
    ferrum_gguf::Gguf::parse(std::fs::read(p).unwrap())
        .unwrap()
        .tensor("output.weight")
        .is_some()
}

/// The final loss in the `steps: N loss a -> b` line.
fn last_loss(out: &str) -> f32 {
    let line = out
        .lines()
        .find(|l| l.starts_with("steps:"))
        .expect("steps line");
    line.split_whitespace()
        .filter_map(|w| w.parse::<f32>().ok())
        .next_back()
        .expect("a loss")
}

fn scratch(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let d = std::env::temp_dir().join(format!("ferrum_gpu_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let data = d.join("c.txt");
    std::fs::write(&data, "the cat sat on the mat. ".repeat(60)).unwrap();
    (d, data)
}

const GPU: &str = "--device gpu --gpu-adapter llvmpipe";

#[test]
fn a_tied_model_trains_on_a_gpu_adapter_and_is_stored_tied() {
    if !have_llvmpipe() {
        eprintln!("SKIP: no llvmpipe adapter");
        return;
    }
    let (d, data) = scratch("tie");
    let (gpu, cpu) = (d.join("g.gguf"), d.join("c.gguf"));
    let out = train(GPU, "--tie", &data, &gpu).unwrap();
    assert!(out.contains("llvmpipe"), "trained on the adapter: {out}");
    train("--device cpu", "--tie", &data, &cpu).unwrap();
    assert!(
        !has_own_head(&gpu),
        "the GPU-trained file stores the head once"
    );
    assert!(!has_own_head(&cpu));
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn dropout_trains_on_a_gpu_adapter_to_the_same_loss_as_the_cpu() {
    if !have_llvmpipe() {
        eprintln!("SKIP: no llvmpipe adapter");
        return;
    }
    let (d, data) = scratch("drop");
    let (gpu, cpu) = (d.join("g.gguf"), d.join("c.gguf"));
    let on_gpu = train(GPU, "--dropout 0.2 --tie", &data, &gpu).unwrap();
    assert!(on_gpu.contains("llvmpipe"), "{on_gpu}");
    let on_cpu = train("--device cpu", "--dropout 0.2 --tie", &data, &cpu).unwrap();
    let (a, b) = (last_loss(&on_gpu), last_loss(&on_cpu));
    assert!(
        (a - b).abs() <= 1e-3 * b.abs().max(1.0),
        "final loss gpu {a} vs cpu {b}"
    );
    // and dropout really was on: the run differs from the same run without it
    let plain = train(GPU, "--tie", &data, &d.join("p.gguf")).unwrap();
    assert_ne!(last_loss(&plain), a);
    std::fs::remove_dir_all(d).ok();
}
