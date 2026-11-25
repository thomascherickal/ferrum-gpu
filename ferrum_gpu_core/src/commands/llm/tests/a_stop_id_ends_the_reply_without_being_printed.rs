use super::*;
use crate::commands::gpt2::gen::run;

#[test]
fn a_stop_id_ends_the_reply_without_being_printed() {
    let tok = {
        use ferrum_gguf::GgufBuilder;
        let mut b = GgufBuilder::new();
        b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
        b.meta("tokenizer.ggml.tokens", arr(&["a", "b", "c", "d"]));
        GgufTokenizer::from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap()
    };
    // The decoder says "a", "a", then "d" (the stop id) forever.
    let mut calls = 0;
    let mut feed = |_t: usize| -> Result<Vec<f32>, String> {
        calls += 1;
        let mut l = vec![0.0f32; 4];
        l[if calls < 3 { 0 } else { 3 }] = 9.0;
        Ok(l)
    };
    let o = crate::commands::gpt2::gen::Opts {
        max_new: 10,
        params: Params::default(),
        seed: 1,
        stops: vec![],
        stream: false,
        stop_ids: vec![3],
    };
    let (text, n) = run(&mut feed, &|ids| tok.decode(ids), 64, &[1], &o, None).unwrap();
    assert_eq!((text.as_str(), n), ("aa", 2));
}

#[test]
fn a_closed_output_is_reported_at_every_write_of_the_imported_model_commands() {
    use crate::commands::gpt2::error_paths::sweep::sweep;
    let (m, data) = model("sweep", &TOKS, true);
    let (p, d) = (m.display(), data.display());
    assert!(
        sweep(&format!(
            "generate --model {p} --prompt ab --max-new 3 --device cpu --stream"
        )) > 0
    );
    assert!(
        sweep(&format!(
            "run-gguf --model {p} --prompt ab --max-new 3 --device cpu"
        )) > 0
    );
    let json = m.with_file_name("e.json");
    assert!(
        sweep(&format!(
            "eval --model {p} --data {d} --json {}",
            json.display()
        )) > 0
    );
}

#[test]
fn a_prompt_longer_than_the_context_and_a_memory_budget_that_is_too_small_are_refused() {
    let (m, _) = model("refuse", &TOKS, false);
    let p = m.display();
    let ids = vec!["0"; 100].join(" ");
    let long: Vec<String> = [
        "generate",
        "--model",
        &p.to_string(),
        "--device",
        "cpu",
        "--ids",
        &ids,
    ]
    .map(String::from)
    .to_vec();
    let e = cli(&long, &mut Vec::new()).unwrap_err();
    assert!(e.contains("context"), "{e}");
    assert!(!err(&format!(
        "generate --model {p} --prompt ab --device gpu --gpu-mem-mb 0"
    ))
    .is_empty());
}
