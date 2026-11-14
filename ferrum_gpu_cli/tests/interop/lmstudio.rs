//! LM Studio: `export-lmstudio` runs the right `lms` commands. The real `lms` registers models under
//! the user's `~/.lmstudio`, outside this repository, so a fake `lms` (a shell script that records
//! what it was asked and what file it was given) stands in; LM Studio's engine is llama.cpp, so that
//! the imported file runs is shown by loading the very bytes `lms` received in llama.cpp.

use crate::skip;
use crate::support::model::{ferrum, fixture, ours, NEW_TOKENS, PROMPT};
use crate::support::proc::with_deadline;
use crate::support::tools::llamacpp_dir;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// A fake `lms` and the files it leaves behind.
struct Fake {
    bin: PathBuf,
    log: PathBuf,
    captured: PathBuf,
}

/// Write a fake `lms` into a fresh directory `tag`. On `import` it records the arguments and copies
/// the model file it was given; on `ls --json` it lists that file under `repo`. `import_exit` is the
/// status `import` returns.
fn fake(tag: &str, repo: &str, file: &str, import_exit: i32) -> Fake {
    let dir = fixture().dir.join(format!("lms-{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    let (log, captured) = (dir.join("calls.log"), dir.join("captured.gguf"));
    let script = format!(
        "#!/bin/sh\n\
         echo \"$@\" >> '{log}'\n\
         case \"$1\" in\n\
         import) for last; do :; done; cp \"$last\" '{cap}'; \
                 [ {exit} -eq 0 ] || {{ echo 'import refused by the fake lms' >&2; exit {exit}; }} ;;\n\
         ls) printf '[{{\"modelKey\":\"fake/{repo}\",\"path\":\"/home/u/.lmstudio/models/{repo}/{file}\"}}]\\n' ;;\n\
         esac\n",
        log = log.display(),
        cap = captured.display(),
        exit = import_exit,
    );
    let bin = dir.join("lms");
    std::fs::write(&bin, script).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    Fake { bin, log, captured }
}

fn export(extra: &[&str], model: &Path, tool: &Path) -> Output {
    let mut args = vec![
        "export-lmstudio",
        "--model",
        model.to_str().unwrap(),
        "--bin",
        tool.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    ferrum(&args)
}

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
    let log = std::fs::read_to_string(&fk.log).unwrap();
    let calls: Vec<&str> = log.lines().collect();
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
    assert_eq!(
        std::fs::read(&fk.captured).unwrap(),
        std::fs::read(&f.q8).unwrap()
    );
    // LM Studio runs GGUF with llama.cpp: the bytes it was given give our text there too
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed: file bytes checked, engine not run")
    };
    let o = with_deadline(&dir.join("llama-completion"), 120)
        .env("LD_LIBRARY_PATH", &dir)
        .args(["-m"])
        .arg(&fk.captured)
        .args([
            "-p",
            PROMPT,
            "-n",
            &NEW_TOKENS.to_string(),
            "--temp",
            "0",
            "-no-cnv",
            "--no-warmup",
            "-ngl",
            "0",
            "-t",
            "1",
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(o.status.success());
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim_end(), ours(&f.q8));
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
