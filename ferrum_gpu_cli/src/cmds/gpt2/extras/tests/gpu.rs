//! `train --tie` on a GPU adapter, through the real command. Uses llvmpipe (the software Vulkan
//! adapter) and skips, printing why, where it does not exist.

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

fn train_tied(device: &str, data: &Path, out: &Path) -> Result<String, String> {
    let mut buf = Vec::new();
    run(
        &words(&format!(
            "train {device} --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 \
             --epochs 3 --vocab 300 --seed 5 --tie --data {} --out {}",
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

#[test]
fn a_tied_model_trains_on_a_gpu_adapter_and_is_stored_tied() {
    if !have_llvmpipe() {
        eprintln!("SKIP: no llvmpipe adapter");
        return;
    }
    let d = std::env::temp_dir().join(format!("ferrum_gputie_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let data = d.join("c.txt");
    std::fs::write(&data, "the cat sat on the mat. ".repeat(60)).unwrap();
    let (gpu, cpu) = (d.join("g.gguf"), d.join("c.gguf"));
    let out = train_tied("--device gpu --gpu-adapter llvmpipe", &data, &gpu).unwrap();
    assert!(out.contains("llvmpipe"), "trained on the adapter: {out}");
    assert!(!out.contains("training on the CPU engine"), "{out}");
    train_tied("--device cpu", &data, &cpu).unwrap();
    assert!(
        !has_own_head(&gpu),
        "the GPU-trained file stores the head once"
    );
    assert!(!has_own_head(&cpu));
    std::fs::remove_dir_all(d).ok();
}
