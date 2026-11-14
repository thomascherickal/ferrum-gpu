//! llama.cpp reads our files and we read its: same token ids, same greedy text, for every precision
//! we write and for the files `llama-quantize` writes.

use crate::skip;
use crate::support::model::{fixture, ours, NEW_TOKENS, PROMPT};
use crate::support::proc::with_deadline;
use crate::support::tools::llamacpp_dir;
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// `llama-completion` on `model`: the prompt then up to [`NEW_TOKENS`] tokens, greedy, CPU only.
fn completion(dir: &Path, model: &Path) -> String {
    let o = with_deadline(&dir.join("llama-completion"), 120)
        .env("LD_LIBRARY_PATH", dir)
        .args(["-m"])
        .arg(model)
        .args(["-p", PROMPT, "-n", &NEW_TOKENS.to_string()])
        .args([
            "--temp",
            "0",
            "-no-cnv",
            "--no-warmup",
            "-ngl",
            "0",
            "-t",
            "1",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("run llama-completion");
    assert!(
        o.status.success(),
        "llama-completion {}: {}",
        model.display(),
        String::from_utf8_lossy(&o.stderr)
            .lines()
            .filter(|l| l.contains("error") || l.contains("failed"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    String::from_utf8_lossy(&o.stdout).trim_end().to_string()
}

/// `llama-tokenize --ids` of `text` with `model`'s vocabulary.
fn tokenize(dir: &Path, model: &Path, text: &str) -> Vec<usize> {
    let o = with_deadline(&dir.join("llama-tokenize"), 60)
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
    let o = with_deadline(&dir.join("llama-quantize"), 120)
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

#[test]
fn our_tokenizer_gives_the_ids_llama_cpp_gives() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    let f = fixture();
    let ours = ferrum_gpu_cli::cmds::gpt2::load(f.f32.to_str().unwrap()).unwrap();
    for text in [
        "the cat sat",
        "the",
        "a",
        " the  cat",
        "the cat sat on the mat. the cat",
        "x y\tz",
        "café au lait",
        "日本語",
    ] {
        assert_eq!(
            tokenize(&dir, &f.f32, text),
            ours.tok.encode(text),
            "ids of {text:?}"
        );
    }
}

#[test]
fn greedy_text_is_the_same_in_llama_cpp_for_every_precision_we_write() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    for (label, path) in fixture().all() {
        let (theirs, mine) = (completion(&dir, path), ours(path));
        assert_eq!(theirs, mine, "{label}");
        assert!(
            mine.starts_with(PROMPT) && mine.len() > PROMPT.len() + 5,
            "{label}: {mine:?}"
        );
    }
}

#[test]
fn files_llama_quantize_writes_load_here_and_say_what_llama_cpp_says() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    let f = fixture();
    let reference = ours(&f.f32);
    // formats we claim to read: the block formats and two k-quants (a tensor whose width is not a
    // multiple of 256 is written in a smaller format by llama-quantize itself)
    for ty in ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K_M", "Q6_K"] {
        let out: PathBuf = f.dir.join(format!("lq-{ty}.gguf"));
        quantize(&dir, &f.f32, &out, ty);
        let (mine, theirs) = (ours(&out), completion(&dir, &out));
        assert_eq!(
            mine, theirs,
            "{ty}: the two readers disagree about one file"
        );
        // and it is still the model: this sentence is learned far past the quantization noise
        assert_eq!(mine, reference, "{ty}: quantizing changed the text");
    }
}

#[test]
fn a_file_re_exported_here_from_a_llama_cpp_file_still_loads_in_llama_cpp() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    let f = fixture();
    let lq = f.dir.join("lq-Q8_0-again.gguf");
    quantize(&dir, &f.f32, &lq, "Q8_0");
    let back = f.dir.join("roundtrip.gguf");
    crate::support::model::ferrum_ok(&[
        "export-gguf",
        "--model",
        lq.to_str().unwrap(),
        "--out",
        back.to_str().unwrap(),
        "--quant",
        "int8",
    ]);
    assert_eq!(completion(&dir, &back), ours(&back));
    assert_eq!(completion(&dir, &back), ours(&f.f32));
}

#[test]
fn a_file_that_is_not_a_model_is_refused_by_both() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    let f = fixture();
    let bytes = std::fs::read(&f.f32).unwrap();
    let cut = f.dir.join("truncated.gguf");
    std::fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
    // ours: a clear error, not a panic
    let o = crate::support::model::ferrum(&[
        "generate",
        "--device",
        "cpu",
        "--model",
        cut.to_str().unwrap(),
        "--prompt",
        "the",
    ]);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!err.contains("panicked"), "{err}");
    // theirs: it refuses too
    let o = with_deadline(&dir.join("llama-completion"), 60)
        .env("LD_LIBRARY_PATH", &dir)
        .args(["-m"])
        .arg(&cut)
        .args([
            "-p",
            "the",
            "-n",
            "2",
            "-no-cnv",
            "--no-warmup",
            "-ngl",
            "0",
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!o.status.success(), "llama.cpp accepted a truncated file");
}
