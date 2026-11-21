use super::*;

#[test]
fn our_tokenizer_gives_the_ids_llama_cpp_gives() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    let f = fixture();
    let ours = ferrum_gpu_core::commands::gpt2::load(f.f32.to_str().unwrap()).unwrap();
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
