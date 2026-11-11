use super::*;
use std::path::PathBuf;

fn a(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn run_ok(s: &str) -> String {
    let mut out = Vec::new();
    run(&a(s), &mut out).unwrap_or_else(|e| panic!("{s}: {e}"));
    String::from_utf8(out).unwrap()
}

fn run_err(s: &str) -> String {
    run(&a(s), &mut Vec::new()).unwrap_err()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn corpus(d: &std::path::Path) -> String {
    let p = d.join("data.txt");
    std::fs::write(&p, "abcabcabcabc ".repeat(40)).unwrap();
    p.display().to_string()
}

#[test]
fn help_and_unknown_commands() {
    assert!(run_ok("help").contains("usage: ferrum-gpu"));
    assert!(run(&[], &mut Vec::new()).unwrap_err().contains("usage"));
    assert!(run_err("frobnicate").contains("unknown command"));
}

#[test]
fn unknown_flags_are_rejected_per_command() {
    assert!(run_err("info --bogus 1").contains("unknown flag"));
    assert!(run_err("eval --device cpu").contains("unknown flag"));
    assert!(
        run_err("train --data x").contains("--out is required")
            || run_err("train --data x").contains("data")
    );
}

#[test]
fn train_generate_run_eval_finetune_round_trip_on_cpu() {
    let d = dir("e2e");
    let data = corpus(&d);
    let m1 = d.join("m1.gguf").display().to_string();
    let m2 = d.join("m2.gguf").display().to_string();
    let t = run_ok(&format!(
        "train --device cpu --preset tiny --data {data} --out {m1} --steps 30 --batch 2 --context 16"
    ));
    assert!(
        t.contains("device: ") && t.contains("steps: 30") && t.contains("wrote "),
        "{t}"
    );
    assert!(run_ok(&format!(
        "generate --device cpu --model {m1} --prompt ab --max-new 5"
    ))
    .starts_with("ab"));
    let r = run_ok(&format!(
        "run-gguf --device cpu --model {m1} --prompt ab --max-new 3"
    ));
    assert!(r.contains("model: vocab 256"), "{r}");
    let e = run_ok(&format!("eval --model {m1} --data {data}"));
    assert!(e.contains("perplexity"), "{e}");
    let f = run_ok(&format!(
        "finetune-gguf --device cpu --model {m1} --data {data} --out {m2} --steps 5 --batch 2 --context 16 --quant q8_0"
    ));
    assert!(f.contains("wrote "), "{f}");
    assert!(run_ok(&format!("info --device cpu --model {m2}")).contains("model: vocab 256"));
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn errors_name_the_problem() {
    let d = dir("err");
    let data = corpus(&d);
    let out = d.join("o.gguf").display().to_string();
    assert!(run_err(&format!(
        "train --device cpu --preset huge --data {data} --out {out}"
    ))
    .contains("unknown preset"));
    assert!(run_err(&format!(
        "train --device cpu --data {data} --out {out} --quant q2"
    ))
    .contains("--quant"));
    assert!(run_err(&format!(
        "train --device cpu --data {data} --out {out} --steps 0"
    ))
    .contains("steps"));
    assert!(run_err(&format!("train --device cpu --data /nope --out {out}")).contains("--data"));
    assert!(run_err("generate --device cpu --model /nope --prompt x").contains("/nope"));
    assert!(
        run_err(&format!("train --device tpu --data {data} --out y")).contains("unknown device")
    );
    let bad = d.join("bad.gguf");
    std::fs::write(&bad, b"not gguf").unwrap();
    assert!(run_err(&format!("eval --model {} --data {data}", bad.display())).contains("magic"));
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn mean_loss_of_untrained_model_is_is_finite_and_sane() {
    let m = fresh_model("tiny", 1).unwrap();
    let (loss, n) = mean_loss(&m, b"hello world, hello world").unwrap();
    assert!(
        n > 0 && loss.is_finite() && loss > 0.0 && loss < 12.0,
        "{loss}"
    );
    assert!(mean_loss(&m, b"x").unwrap_err().contains("at least 2"));
}
