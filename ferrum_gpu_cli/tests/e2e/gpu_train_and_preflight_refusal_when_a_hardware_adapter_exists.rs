use super::*;

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
