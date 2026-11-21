use super::*;

#[test]
fn with_a_serve_time_the_server_exits_normally() {
    let dir = work("holds");
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
            "--port",
            "0",
        ])
        .env("FERRUM_SERVE_SECONDS", "1")
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&o.stdout).contains("ferrum-gpu server: http://"));
}
