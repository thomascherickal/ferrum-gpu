//! `train --dropout` / `--tie` through the real command: the file written, and what it says.

use crate::cmds::run;
use std::path::{Path, PathBuf};

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn train(extra: &str, data: &Path, out: &Path) -> Result<String, String> {
    let mut buf = Vec::new();
    run(
        &words(&format!(
            "train --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 \
             --epochs 3 --vocab 300 --seed 5 --data {} --out {} {extra}",
            data.display(),
            out.display()
        )),
        &mut buf,
    )?;
    Ok(String::from_utf8(buf).unwrap())
}

fn scratch(name: &str) -> (PathBuf, PathBuf) {
    let d = std::env::temp_dir().join(format!("ferrum_extras_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let data = d.join("c.txt");
    std::fs::write(&data, "the cat sat on the mat. ".repeat(60)).unwrap();
    (d, data)
}

fn bytes(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap()
}

#[test]
fn dropout_zero_writes_the_same_file_as_no_flag() {
    let (d, data) = scratch("zero");
    let (a, b) = (d.join("a.gguf"), d.join("b.gguf"));
    train("", &data, &a).unwrap();
    train("--dropout 0", &data, &b).unwrap();
    assert_eq!(bytes(&a), bytes(&b));
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn dropout_trains_on_the_cpu_and_changes_the_model() {
    let (d, data) = scratch("on");
    let (plain, dropped) = (d.join("p.gguf"), d.join("d.gguf"));
    train("", &data, &plain).unwrap();
    let out = train("--dropout 0.2", &data, &dropped).unwrap();
    assert!(
        out.contains("--dropout: training on the CPU engine"),
        "{out}"
    );
    assert!(out.contains("steps:"), "{out}");
    assert_ne!(
        bytes(&plain),
        bytes(&dropped),
        "dropout changes the weights"
    );
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn the_same_seed_writes_the_same_dropout_model_byte_for_byte() {
    let (d, data) = scratch("repeat");
    let (a, b) = (d.join("a.gguf"), d.join("b.gguf"));
    train("--dropout 0.2", &data, &a).unwrap();
    train("--dropout 0.2", &data, &b).unwrap();
    assert_eq!(bytes(&a), bytes(&b));
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn dropout_composes_with_a_tied_head() {
    let (d, data) = scratch("both");
    let m = d.join("m.gguf");
    let out = train("--dropout 0.2 --tie", &data, &m).unwrap();
    assert!(
        out.contains("--tie and --dropout: training on the CPU engine"),
        "{out}"
    );
    let g = ferrum_gguf::Gguf::parse(bytes(&m)).unwrap();
    assert!(
        g.tensor("output.weight").is_none(),
        "the head is stored once"
    );
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn an_explicit_gpu_is_refused_before_any_work() {
    let (d, data) = scratch("gpu");
    let m = d.join("m.gguf");
    let e = {
        let mut buf = Vec::new();
        run(
            &words(&format!(
                "train --device gpu --dropout 0.1 --data {} --out {}",
                data.display(),
                m.display()
            )),
            &mut buf,
        )
        .unwrap_err()
    };
    assert!(
        e.contains("--dropout") && e.contains("CPU engine only"),
        "{e}"
    );
    assert!(!m.exists());
    std::fs::remove_dir_all(d).ok();
}
