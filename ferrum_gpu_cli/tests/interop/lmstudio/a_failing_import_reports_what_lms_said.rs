use super::*;

#[test]
fn a_failing_import_reports_what_lms_said() {
    let f = fixture();
    let fk = fake("fail", "ferrum/f32", "f32.gguf", 3);
    let o = export(&[], &f.f32, &fk.bin);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("`lms import` failed") && err.contains("import refused by the fake lms"),
        "{err}"
    );
    // and it did not go on to list
    let log = std::fs::read_to_string(&fk.log).unwrap();
    assert_eq!(log.lines().count(), 1, "{log}");
}

#[test]
fn an_architecture_that_cannot_run_from_lms_import_is_refused_by_name() {
    let f = fixture();
    let llama = f.dir.join("llama-style.gguf");
    crate::support::model::ferrum_ok(&[
        "train",
        "--device",
        "cpu",
        "--arch",
        "llama",
        "--preset",
        "tiny",
        "--steps",
        "2",
        "--batch",
        "2",
        "--context",
        "16",
        "--data",
        f.corpus.to_str().unwrap(),
        "--out",
        llama.to_str().unwrap(),
    ]);
    let fk = fake("arch", "ferrum/l", "llama-style.gguf", 0);
    let o = export(&[], &llama, &fk.bin);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("architecture 'llama' cannot be exported"),
        "{err}"
    );
    assert!(!fk.log.exists());
}

#[test]
fn a_missing_lms_binary_is_named() {
    let f = fixture();
    let o = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu"))
        .args(["export-lmstudio", "--model"])
        .arg(&f.f32)
        .args(["--bin", "/nonexistent/lms"])
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--bin /nonexistent/lms: no such file"));
}
