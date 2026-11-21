//! The formats added after Q4_0 (F16, MXFP4, Q2_K): the files we write have the ggml types
//! llama.cpp expects, llama.cpp reads them and continues the prompt exactly as we do, and our
//! Q2_K encoder is as accurate as `llama-quantize`'s.
use crate::skip;
use crate::support::llama::completion;
use crate::support::model::{ours, PROMPT};
use crate::support::proc::with_deadline;
use crate::support::tools::llamacpp_dir;
use crate::support::wide::wide;
use ferrum_gguf::Gguf;
use std::process::Stdio;

const F16: u32 = 1;
const Q2_K: u32 = 10;
const MXFP4: u32 = 39;

fn load(p: &std::path::Path) -> Gguf {
    Gguf::parse(std::fs::read(p).unwrap()).unwrap()
}

/// The ggml type of every 2-D weight of the file (the position embedding is always f32) and of every
/// vector.
fn types(g: &Gguf) -> (Vec<u32>, Vec<u32>) {
    let (mut mats, mut vecs) = (vec![], vec![]);
    for t in &g.tensors {
        if t.name.starts_with("position_embd") {
            continue;
        }
        if t.dims.len() >= 2 {
            mats.push(t.ggml_type);
        } else {
            vecs.push(t.ggml_type);
        }
    }
    (mats, vecs)
}

fn rmse(a: &[f32], b: &[f32]) -> f64 {
    let s: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    (s / a.len() as f64).sqrt()
}

mod each_precision_writes_the_ggml_type_llama_cpp_expects;
