use super::*;

#[test]
fn train_writes_a_gpt2_file_with_its_tokenizer_and_generate_uses_it() {
    let d = dir("roundtrip");
    let (c, m, tok) = (data(&d), d.join("m.gguf"), d.join("tok.gguf"));
    let t = ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --vocab 300 --tokenizer {}",
        m.display(),
        tok.display()
    ));
    assert!(
        t.contains("tokenizer: trained") && t.contains("steps: 21") && t.contains("wrote "),
        "{t}"
    );
    assert!(is_gpt2(m.to_str().unwrap()) && !is_gpt2("/nope"));
    // a second run reuses the saved tokenizer file
    let t2 = ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --tokenizer {}",
        m.display(),
        tok.display()
    ));
    assert!(
        t2.contains("tokenizer: ") && !t2.contains("trained"),
        "{t2}"
    );
    let g = ok(&format!(
        "generate --device cpu --model {} --prompt the --max-new 8",
        m.display()
    ));
    assert!(g.starts_with("the"), "{g}");
    assert!(ok(&format!("info --device cpu --model {}", m.display())).contains("model: gpt2 vocab"));
    let r = ok(&format!(
        "run-gguf --device cpu --model {} --prompt the --max-new 2",
        m.display()
    ));
    assert!(
        r.contains("device: CPU") && r.contains("model: gpt2"),
        "{r}"
    );
}
