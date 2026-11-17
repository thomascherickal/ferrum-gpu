use crate::jobs::allow::{stop_file, validate, COMMANDS, MAX_ARGS, MAX_ARG_BYTES};
use std::path::PathBuf;

fn v(a: &[&str]) -> Vec<String> {
    a.iter().map(|s| s.to_string()).collect()
}

#[test]
fn every_listed_command_is_accepted_and_others_are_refused() {
    for c in COMMANDS {
        assert!(validate(c, &v(&["--x", "y"])).is_ok(), "{c}");
    }
    for c in [
        "",
        "sh",
        "bash",
        "rm",
        "ferrum-gpu",
        "train ",
        "../train",
        "--help",
    ] {
        let e = validate(c, &[]).unwrap_err();
        assert!(e.contains("not a command"), "{c:?}: {e}");
    }
}

#[test]
fn argument_limits_are_enforced() {
    assert!(validate("info", &vec!["a".to_string(); MAX_ARGS]).is_ok());
    assert!(validate("info", &vec!["a".to_string(); MAX_ARGS + 1]).is_err());
    assert!(validate("info", &["a".repeat(MAX_ARG_BYTES)]).is_ok());
    assert!(validate("info", &["a".repeat(MAX_ARG_BYTES + 1)]).is_err());
    assert!(validate("info", &v(&["bad\0arg"]))
        .unwrap_err()
        .contains("NUL"));
}

#[test]
fn bin_may_only_name_ollama_or_lms() {
    for ok in [
        "ollama",
        "/usr/local/bin/ollama",
        "lms",
        "C:/x/lms.exe",
        "ollama.exe",
    ] {
        assert!(
            validate("export-ollama", &v(&["--bin", ok])).is_ok(),
            "{ok}"
        );
    }
    for bad in ["sh", "/bin/sh", "/tmp/ollama2", "ollama/../sh", "", "-c"] {
        assert!(
            validate("export-ollama", &v(&["--bin", bad])).is_err(),
            "{bad:?}"
        );
    }
    assert!(validate("export-ollama", &v(&["--model", "m.gguf", "--bin"])).is_err());
    // the value is judged only when it follows --bin
    assert!(validate("export-ollama", &v(&["--name", "sh"])).is_ok());
}

#[test]
fn a_training_command_has_a_stop_file_next_to_its_output() {
    let args = v(&["--data", "t.txt", "--out", "/tmp/m.gguf"]);
    assert_eq!(
        stop_file("train", &args),
        Some(PathBuf::from("/tmp/m.stop"))
    );
    assert_eq!(
        stop_file("finetune-gguf", &args),
        Some(PathBuf::from("/tmp/m.stop"))
    );
    assert_eq!(
        stop_file("run", &v(&["--model", "d/r.gguf", "--data", "x"])),
        Some(PathBuf::from("d/r.stop"))
    );
    assert_eq!(stop_file("generate", &args), None);
    assert_eq!(stop_file("train", &v(&["--data", "t"])), None);
    assert_eq!(stop_file("train", &v(&["--out"])), None);
}
