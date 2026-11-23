use super::*;
use crate::commands::gpt2::error_paths::sweep::sweep;
use ferrum_gguf::GgufBuilder;

#[test]
fn a_file_that_is_not_a_llama_model_is_named() {
    let (m, _) = model("notllama", &TOKS, false);
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("mamba".into()));
    b.meta("tokenizer.ggml.tokens", arr(&TOKS));
    let other = m.with_file_name("mamba.gguf");
    std::fs::write(&other, b.into_bytes()).unwrap();
    let e = load(&other.display().to_string()).err().unwrap();
    assert!(e.starts_with(&format!("{}: ", other.display())), "{e}");
    assert!(from_bytes(b"not gguf".to_vec())
        .unwrap_err()
        .contains("magic"));
    assert!(from_bytes(std::fs::read(&other).unwrap())
        .unwrap_err()
        .contains("mamba"));
}

#[test]
fn a_llama_file_with_a_broken_vocabulary_is_named() {
    let (m, _) = model("brokentok", &TOKS, false);
    let cfg = preset("tiny").unwrap();
    let llama = Llama::random(
        ferrum_ref::llama::Config { vocab: 12, ..cfg },
        3,
        false,
        false,
    );
    // a token list without the `tokenizer.ggml.model` key that says how to read it
    let meta = vec![("tokenizer.ggml.tokens".to_string(), arr(&TOKS))];
    let p = m.with_file_name("novocabkind.gguf");
    std::fs::write(&p, to_gguf_with_meta(&llama, Quant::F32, &meta)).unwrap();
    let e = load(&p.display().to_string()).err().unwrap();
    assert!(e.starts_with(&format!("{}: ", p.display())), "{e}");
}

#[test]
fn a_report_file_that_cannot_be_written_is_named() {
    let (m, data) = model("badjson", &TOKS, false);
    let json = "/nonexistent-ferrum-dir/e.json";
    let e = err(&format!(
        "eval --model {} --data {} --json {json}",
        m.display(),
        data.display()
    ));
    assert!(e.starts_with(&format!("{json}: ")), "{e}");
}

#[test]
fn generate_reports_a_closed_output_at_every_write() {
    let (m, _) = model("closed", &TOKS, false);
    let cmd = format!(
        "generate --device cpu --model {} --prompt ab --max-new 3 --stream",
        m.display()
    );
    assert!(sweep(&cmd) > 0);
}
