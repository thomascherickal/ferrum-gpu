use super::*;

#[test]
fn a_file_that_is_not_a_model_is_refused_by_both() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    let f = fixture();
    let bytes = std::fs::read(&f.f32).unwrap();
    let cut = f.dir.join("truncated.gguf");
    std::fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
    // ours: a clear error, not a panic
    let o = crate::support::model::ferrum(&[
        "generate",
        "--device",
        "cpu",
        "--model",
        cut.to_str().unwrap(),
        "--prompt",
        "the",
    ]);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!err.contains("panicked"), "{err}");
    // theirs: it refuses too
    let o = with_deadline(&dir.join("llama-completion"), 60)
        .env("LD_LIBRARY_PATH", &dir)
        .args(["-m"])
        .arg(&cut)
        .args([
            "-p",
            "the",
            "-n",
            "2",
            "-no-cnv",
            "--no-warmup",
            "-ngl",
            "0",
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!o.status.success(), "llama.cpp accepted a truncated file");
}
