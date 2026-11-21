use super::*;

#[test]
fn training_from_a_token_file_equals_training_from_the_text() {
    let d = dir("tokentrain");
    let (c, tok, toks) = (data(&d), d.join("tok.gguf"), d.join("t.tok.gguf"));
    let (from_text, from_tokens) = (d.join("a.gguf"), d.join("b.gguf"));
    let common = "--device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 --epochs 3 --quant f32";
    ok(&format!(
        "train {common} --data {c} --tokenizer {} --out {}",
        tok.display(),
        from_text.display()
    ));
    ok(&format!(
        "tokenize --data {c} --tokenizer {} --out {}",
        tok.display(),
        toks.display()
    ));
    let r = ok(&format!(
        "train {common} --data {} --tokenizer {} --out {}",
        toks.display(),
        tok.display(),
        from_tokens.display()
    ));
    assert!(r.contains("tokenizer: ") && !r.contains("trained"), "{r}");
    assert_eq!(
        std::fs::read(&from_text).unwrap(),
        std::fs::read(&from_tokens).unwrap()
    );
    // validation and --sample work on a token file too
    let v = ok(&format!(
        "train {common} --val 0.2 --sample --data {} --tokenizer {} --out {}",
        toks.display(),
        tok.display(),
        from_tokens.display()
    ));
    assert!(v.contains("val loss") && v.contains("sample: "), "{v}");
}

#[test]
fn a_token_file_needs_the_tokenizer_it_was_made_with() {
    let d = dir("tokenmismatch");
    let (c, tok, other, toks) = (
        data(&d),
        d.join("tok.gguf"),
        d.join("other.gguf"),
        d.join("t.tok.gguf"),
    );
    let c2 = d.join("c2.txt");
    std::fs::write(&c2, "dog dig dug dig dog bog bug. ".repeat(60)).unwrap();
    let small = "--device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --epochs 1";
    // two different tokenizer files, each made by training on different text
    ok(&format!(
        "train {small} --data {c} --tokenizer {} --out {}",
        tok.display(),
        d.join("a.gguf").display()
    ));
    ok(&format!(
        "train {small} --data {} --tokenizer {} --out {}",
        c2.display(),
        other.display(),
        d.join("b.gguf").display()
    ));
    ok(&format!(
        "tokenize --data {c} --tokenizer {} --out {}",
        tok.display(),
        toks.display()
    ));
    let train = |tokenizer: &str| {
        err(&format!(
            "train {small} --data {} {tokenizer} --out {}",
            toks.display(),
            d.join("x.gguf").display()
        ))
    };
    assert!(train("").contains("needs --tokenizer"));
    assert!(
        train(&format!("--tokenizer {}", d.join("missing.gguf").display()))
            .contains("needs --tokenizer")
    );
    assert!(train(&format!("--tokenizer {}", other.display()))
        .contains("not the tokenizer this token file was made with"));
}
