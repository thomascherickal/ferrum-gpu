use super::*;
use ferrum_tok::ByteBpeTokenizer;

// Golden files made by ferrum-cpu (commit 56a9d3a, see fixtures/readme.md).
const TOK: &[u8] = include_bytes!("../../../../fixtures/bpe/tok.gguf");
const MODEL: &[u8] = include_bytes!("../../../../fixtures/bpe/model.gguf");
const CORPUS: &str = include_str!("../../../../fixtures/bpe/corpus.txt");

fn small() -> ByteBpeTokenizer {
    ByteBpeTokenizer::train(&"abab cdcd ünï low lower ".repeat(10), 300).unwrap()
}

mod a_model_written_with_a_tokenizer_reads_back_with_it;
mod arrays_of_different_lengths_are_refused;
mod rejects_missing_wrong_and_disagreeing_keys;
mod round_trips_through_a_tokenizer_file;
