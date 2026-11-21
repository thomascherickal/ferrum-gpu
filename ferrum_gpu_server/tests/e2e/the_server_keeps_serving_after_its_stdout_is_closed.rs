use super::*;

/// Regression (found by CI): the server printed an `auth:` line after its address line, the test's
/// reader had already stopped, and `println!` panicked on the broken pipe, killing the server.
#[test]
fn the_server_keeps_serving_after_its_stdout_is_closed() {
    let dir = work("closed_stdout");
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    let s = spawn(model.to_str().unwrap(), &["--token", "s3cret"], &[]);
    // `spawn` returned, so the reader is gone; give the server time to print its remaining lines
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert!(http(&s.addr, "GET", "/health", "", "").contains("200"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn exposing_the_server_without_a_token_is_refused() {
    let dir = work("expose");
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu-server"))
        .args([
            "--model",
            model.to_str().unwrap(),
            "--device",
            "cpu",
            "--host",
            "0.0.0.0",
            "--port",
            "0",
        ])
        .env_remove("FERRUM_TOKEN")
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        !o.status.success() && err.contains("not a loopback address") && err.contains("--token"),
        "{err}"
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_missing_model_exits_with_an_error() {
    let o = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu-server"))
        .args(["--model", "/nope.gguf", "--device", "cpu"])
        .output()
        .unwrap();
    assert!(!o.status.success() && String::from_utf8_lossy(&o.stderr).contains("/nope.gguf"));
}

#[test]
fn help_prints_usage() {
    let o = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu-server"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(
        o.status.success()
            && String::from_utf8_lossy(&o.stdout).contains("usage: ferrum-gpu-server")
    );
}
