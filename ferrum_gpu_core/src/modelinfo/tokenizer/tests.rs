use super::*;
use ferrum_gguf::GgufBuilder;

fn arr(items: &[&str]) -> Value {
    Value::Arr(items.iter().map(|s| Value::Str((*s).into())).collect())
}

fn gguf(arch: &str, model: &str, tokens: Option<&[&str]>, template: Option<&str>) -> Gguf {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(arch.into()));
    b.meta("tokenizer.ggml.model", Value::Str(model.into()));
    if let Some(t) = tokens {
        b.meta("tokenizer.ggml.tokens", arr(t));
    }
    if let Some(t) = template {
        b.meta("tokenizer.chat_template", Value::Str(t.into()));
    }
    Gguf::parse(b.into_bytes()).unwrap()
}

#[test]
fn a_chat_template_names_the_format_of_an_instruct_model() {
    let g = gguf(
        "qwen2",
        "gpt2",
        Some(&["a", "<|im_start|>"]),
        Some("{{ '<|im_start|>' }}"),
    );
    let t = tokenizer_info(&g).unwrap();
    assert_eq!(
        (t.model.as_str(), t.tokens, t.chat_format),
        ("gpt2", 2, Some("ChatML"))
    );
}

#[test]
fn an_instruct_vocabulary_without_a_template_is_recognised_by_its_architecture() {
    let g = gguf("llama", "gpt2", Some(&["<|start_header_id|>", "a"]), None);
    assert_eq!(tokenizer_info(&g).unwrap().chat_format, Some("Llama-3"));
}

#[test]
fn a_base_model_has_no_chat_format() {
    let g = gguf("qwen2", "gpt2", Some(&["a", "b", "c"]), None);
    let t = tokenizer_info(&g).unwrap();
    assert_eq!((t.tokens, t.chat_format), (3, None));
}

#[test]
fn a_file_without_a_vocabulary_has_no_tokenizer_info() {
    assert_eq!(tokenizer_info(&gguf("gpt2", "gpt2", None, None)), None);
}

#[test]
fn every_chat_format_has_its_own_name() {
    let names: Vec<_> = [
        ChatFormat::ChatMl,
        ChatFormat::Llama3,
        ChatFormat::Llama2Inst,
        ChatFormat::Zephyr,
    ]
    .map(format_name)
    .to_vec();
    assert_eq!(names, ["ChatML", "Llama-3", "[INST]", "Zephyr"]);
}
