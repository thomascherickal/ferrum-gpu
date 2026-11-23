use super::*;
use ferrum_gguf::{GgufBuilder, Value};

const LLAMA: &str = "--arch llama --device cpu --preset tiny --steps 2 --batch 2 --context 8";

#[test]
fn help_reports_a_closed_output() {
    // a zero-length slice accepts no bytes: the first write fails like a closed pipe
    let e = run(&a("help"), &mut &mut [0u8; 0][..]).unwrap_err();
    assert!(e.contains("failed to write"), "{e}");
}

#[test]
fn a_model_file_that_cannot_be_read_parsed_or_built_is_named() {
    let d = dir("gate_load");
    let text = corpus(&d);
    let mut mamba = GgufBuilder::new();
    mamba.meta("general.architecture", Value::Str("mamba".into()));
    let other = d.join("mamba.gguf");
    std::fs::write(&other, mamba.into_bytes()).unwrap();
    for (path, why) in [
        (d.display().to_string(), ""),
        (text.clone(), "magic"),
        (other.display().to_string(), "mamba"),
    ] {
        let e = gate::load(&path).err().unwrap();
        assert!(
            e.starts_with(&format!("{path}: ")) && e.contains(why),
            "{e}"
        );
    }
}

#[test]
fn llama_commands_name_the_data_and_outputs_they_cannot_use() {
    let d = dir("llama_io");
    let (text, m) = (corpus(&d), d.join("l.gguf"));
    run_ok(&format!(
        "train {LLAMA} --data {text} --out {}",
        m.display()
    ));
    let missing = "/nonexistent-ferrum-dir/x";
    let e = run_err(&format!(
        "train {LLAMA} --data {missing} --out {}",
        m.display()
    ));
    assert!(e.starts_with("--data: "), "{e}");
    let e = run_err(&format!("train {LLAMA} --data {text} --out {missing}"));
    assert!(e.starts_with(&format!("{missing}: ")), "{e}");
    let e = run_err(&format!("eval --model {} --data {missing}", m.display()));
    assert!(e.starts_with("--data: "), "{e}");
    let e = run_err(&format!(
        "export-gguf --model {} --out {missing}",
        m.display()
    ));
    assert!(e.starts_with(&format!("{missing}: ")), "{e}");
}
