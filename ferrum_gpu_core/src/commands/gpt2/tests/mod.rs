use super::*;
use crate::commands::run;
use std::path::PathBuf;

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn ok(s: &str) -> String {
    let mut out = Vec::new();
    run(&words(s), &mut out).unwrap_or_else(|e| panic!("{s}: {e}"));
    String::from_utf8(out).unwrap()
}

fn err(s: &str) -> String {
    run(&words(s), &mut Vec::new()).unwrap_err()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_gpt2_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn data(d: &std::path::Path) -> String {
    let p = d.join("c.txt");
    std::fs::write(&p, "the cat sat on the mat. ".repeat(60)).unwrap();
    p.display().to_string()
}

const FAST: &str =
    "--device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4";

const FIX_TOK: &[u8] = include_bytes!("../../../../../fixtures/bpe/tok.gguf");
const FIX_CORPUS: &[u8] = include_bytes!("../../../../../fixtures/bpe/corpus.txt");
const FIX_TOKENS: &[u8] = include_bytes!("../../../../../fixtures/bpe/corpus.tok.gguf");

mod a_file_without_a_tokenizer_but_with_256_tokens_uses_bytes;
mod sizes_follow_ferrum_cpu_defaults_presets_and_explicit_flags;
mod the_files_of_a_tokenizer_without_end_of_text_are_joined_with_no_separator_token;
mod tokenize_reads_directories_custom_separators_and_refuses_bad_input;
mod train_writes_a_gpt2_file_with_its_tokenizer_and_generate_uses_it;
mod training_from_a_token_file_equals_training_from_the_text;
mod validation_keeps_the_best_epoch_and_can_stop_early;
