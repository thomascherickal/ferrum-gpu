use super::*;

fn w(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn err(s: &str) -> String {
    split(&w(s)).unwrap_err()
}

#[test]
fn train_only_flags_go_to_train_and_generate_only_flags_to_generate() {
    let s = split(&w(
        "--data c.txt --model m.gguf --prompt hi --epochs 3 --max-new 5 --raw",
    ))
    .unwrap();
    assert_eq!(s.model, "m.gguf");
    assert_eq!(s.train, w("--data c.txt --epochs 3"));
    assert_eq!(s.generate, w("--prompt hi --max-new 5 --raw"));
}

#[test]
fn shared_flags_go_to_both_halves() {
    let s = split(&w("--model m --device cpu --threads 2 --seed 7")).unwrap();
    assert_eq!(s.train, w("--device cpu --threads 2 --seed 7"));
    assert_eq!(s.generate, w("--device cpu --threads 2 --seed 7"));
}

#[test]
fn the_inline_form_is_passed_on_as_two_tokens() {
    let s = split(&w("--model m --seed=7 --prompt=hi")).unwrap();
    assert_eq!(s.train, w("--seed 7"));
    assert_eq!(s.generate, w("--seed 7 --prompt hi"));
}

#[test]
fn a_repeatable_flag_keeps_every_occurrence_in_order() {
    let s = split(&w("--model m --prompt a --stop x --stop y")).unwrap();
    assert_eq!(s.generate, w("--prompt a --stop x --stop y"));
    assert!(s.train.is_empty());
}

#[test]
fn the_model_flag_is_consumed_and_never_forwarded() {
    let s = split(&w("--model m.gguf")).unwrap();
    assert_eq!(s.model, "m.gguf");
    assert!(s.train.is_empty() && s.generate.is_empty());
}

#[test]
fn out_is_refused_in_favour_of_model() {
    assert!(err("--model m --out o").contains("--model for the model file"));
}

#[test]
fn unknown_flags_and_stray_words_are_named() {
    assert!(err("--model m --bogus 1").contains("unknown flag '--bogus'"));
    assert!(err("--model m stray").contains("unknown flag 'stray'"));
}

#[test]
fn model_is_required_and_cannot_repeat() {
    assert!(err("--prompt hi").contains("--model is required"));
    assert!(err("--model m --model n").contains("given twice"));
}

#[test]
fn value_arity_is_checked() {
    assert!(err("--model m --epochs").contains("needs a value"));
    assert!(err("--model m --raw=1").contains("takes no value"));
}
