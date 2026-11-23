use super::*;
use crate::sha256::sha256_hex;
use crate::tokenizer_gguf::from_gguf_bytes;
use ferrum_ref::trainer::TokenSource;
use std::path::PathBuf;

const TOK: &[u8] = include_bytes!("../../../../fixtures/bpe/tok.gguf");
const CORPUS: &str = include_str!("../../../../fixtures/bpe/corpus.txt");
const THEIRS: &[u8] = include_bytes!("../../../../fixtures/bpe/corpus.tok.gguf");

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("ferrum_tokfile_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).display().to_string()
}

fn meta(vocab: usize, eos: Option<u32>) -> TokenFileMeta {
    TokenFileMeta {
        vocab_size: vocab,
        eos_id: eos,
        tokenizer_sha256: "ab".repeat(32),
    }
}

fn ids(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 31 + 7) % vocab) as u32).collect()
}

fn header_bytes_len(m: &TokenFileMeta, n: u64, d: u64) -> usize {
    super::writer::header_bytes(m, n, d).len()
}

mod an_empty_file_has_no_documents_and_no_eos_means_one_document_per_file;
mod io_failures_are_reported_not_panicked;
mod writing_the_fixture_corpus_reproduces_ferrum_cpus_file_byte_for_byte;
