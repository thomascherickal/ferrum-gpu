//! `run` against a real tiny model: train, generate, then load without retraining.

use super::w;
use std::path::PathBuf;

fn go(s: &str) -> Result<String, String> {
    let mut out = Vec::new();
    crate::cmds::run(&w(s), &mut out)?;
    Ok(String::from_utf8(out).unwrap())
}

/// A scratch directory holding a corpus, and the command line for a fast CPU run in it.
struct Case {
    dir: PathBuf,
    model: PathBuf,
    command: String,
}

fn case(name: &str) -> Case {
    let dir = std::env::temp_dir().join(format!("ferrum_run_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("c.txt");
    std::fs::write(&corpus, "the cat sat on the mat. ".repeat(60)).unwrap();
    let model = dir.join("m.gguf");
    let command = format!(
        "run --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 \
         --epochs 2 --vocab 300 --data {} --model {} --prompt the --max-new 6",
        corpus.display(),
        model.display()
    );
    Case {
        dir,
        model,
        command,
    }
}

fn last_line(s: &str) -> &str {
    s.trim_end().lines().last().unwrap()
}

#[test]
fn a_missing_model_is_trained_then_generated_from() {
    let c = case("trains");
    let out = go(&c.command).unwrap();
    assert!(out.contains("steps:") && out.contains("wrote "), "{out}");
    assert!(last_line(&out).starts_with("the"), "{out}");
    assert!(c.model.exists());
    std::fs::remove_dir_all(c.dir).ok();
}

#[test]
fn an_existing_model_is_loaded_not_retrained() {
    let c = case("loads");
    go(&c.command).unwrap();
    let again = go(&c.command).unwrap();
    assert!(again.contains("exists: loading it"), "{again}");
    assert!(!again.contains("steps:"), "{again}");
    assert!(last_line(&again).starts_with("the"), "{again}");
    std::fs::remove_dir_all(c.dir).ok();
}

#[test]
fn a_missing_prompt_is_refused_before_anything_is_trained() {
    let c = case("noprompt");
    let e = go(&c.command.replace(" --prompt the", "")).unwrap_err();
    assert!(e.contains("needs --prompt"), "{e}");
    assert!(!c.model.exists(), "nothing was trained");
    std::fs::remove_dir_all(c.dir).ok();
}
