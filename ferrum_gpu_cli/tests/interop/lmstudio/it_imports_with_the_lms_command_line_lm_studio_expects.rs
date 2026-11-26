use super::*;

#[test]
fn it_imports_with_the_lms_command_line_lm_studio_expects() {
    let f = fixture();
    let fk = fake("ok", "ferrum/f32", "f32.gguf", 0);
    let o = export(&[], &f.f32, &fk.bin);
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(
        o.status.success(),
        "{out}{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        out.contains("imported into LM Studio as 'fake/ferrum/f32'"),
        "{out}"
    );
    assert!(out.contains("lms load fake/ferrum/f32"), "{out}");
    let calls = logged_calls(&fk);
    let abs = std::path::absolute(&f.f32).unwrap();
    assert_eq!(
        calls[0],
        format!("import -y -l --user-repo ferrum/f32 {}", abs.display())
    );
    assert_eq!(calls[1], "ls --json");
}

#[test]
fn the_file_lms_receives_is_exactly_the_model_and_runs_in_llama_cpp() {
    let f = fixture();
    let fk = fake("bytes", "ferrum/q8", "q8.gguf", 0);
    assert!(export(&[], &f.q8, &fk.bin).status.success());
    let received = imported_path(&fk);
    assert_eq!(
        std::fs::read(&received).unwrap(),
        std::fs::read(&f.q8).unwrap()
    );
    // LM Studio runs GGUF with llama.cpp: the bytes it was given give our text there too
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed: file bytes checked, engine not run")
    };
    assert_eq!(completion(&dir, &received), ours(&f.q8));
}

#[test]
fn a_custom_repo_name_is_used() {
    let f = fixture();
    let fk = fake("repo", "me/mytiny", "f32.gguf", 0);
    let o = export(&["--repo", "me/mytiny"], &f.f32, &fk.bin);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let log = std::fs::read_to_string(&fk.log).unwrap();
    assert!(
        log.lines()
            .next()
            .unwrap()
            .contains("--user-repo me/mytiny"),
        "{log}"
    );
}

#[test]
fn a_repo_that_is_not_publisher_slash_name_is_refused_before_lms_runs() {
    let f = fixture();
    for bad in ["nopublisher", "a/b/c", "/name", "pub/"] {
        let fk = fake(&format!("bad{}", bad.len()), "x/y", "f32.gguf", 0);
        let o = export(&["--repo", bad], &f.f32, &fk.bin);
        assert!(!o.status.success(), "{bad}");
        assert!(
            String::from_utf8_lossy(&o.stderr).contains("must look like <publisher>/<name>"),
            "{bad}"
        );
        assert!(!fk.log.exists(), "{bad}: lms must not have been run");
    }
}
