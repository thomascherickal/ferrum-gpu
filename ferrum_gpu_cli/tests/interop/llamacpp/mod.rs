//! llama.cpp reads our files and we read its: same token ids, same greedy text, for every precision
//! we write and for the files `llama-quantize` writes.
use crate::skip;
use crate::support::llama::completion;
use crate::support::model::{fixture, ours, PROMPT};
use crate::support::proc::with_deadline;
use crate::support::tools::llamacpp_dir;
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// `llama-tokenize --ids` of `text` with `model`'s vocabulary.
fn tokenize(dir: &Path, model: &Path, text: &str) -> Vec<usize> {
    let o = with_deadline(dir.join("llama-tokenize"), 60)
        .env("LD_LIBRARY_PATH", dir)
        .args(["-m"])
        .arg(model)
        .args(["-p", text, "--ids", "--log-disable"])
        .stdin(Stdio::null())
        .output()
        .expect("run llama-tokenize");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let out = String::from_utf8_lossy(&o.stdout);
    let inner = out
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_string();
    inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse()
                .unwrap_or_else(|_| panic!("ids in {out:?}"))
        })
        .collect()
}

/// `llama-quantize in out TYPE`.
fn quantize(dir: &Path, input: &Path, output: &Path, ty: &str) {
    let o = with_deadline(dir.join("llama-quantize"), 120)
        .env("LD_LIBRARY_PATH", dir)
        .arg(input)
        .arg(output)
        .arg(ty)
        .stdin(Stdio::null())
        .output()
        .expect("run llama-quantize");
    assert!(
        o.status.success() && output.is_file(),
        "llama-quantize {ty}: {}",
        String::from_utf8_lossy(&o.stderr)
            .lines()
            .last()
            .unwrap_or("")
    );
}

mod a_file_that_is_not_a_model_is_refused_by_both;
mod our_tokenizer_gives_the_ids_llama_cpp_gives;
