use super::*;

#[test]
fn a_file_without_a_tokenizer_but_with_256_tokens_uses_bytes() {
    use ferrum_model::gpt2_gguf::to_gguf;
    use ferrum_ref::gpt2::{Config, Gpt2};
    let d = dir("bytes");
    let cfg = Config {
        vocab: 256,
        ctx: 16,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 16,
    };
    let p = d.join("b.gguf");
    std::fs::write(&p, to_gguf(&Gpt2::init(cfg, 1), Quant::F32)).unwrap();
    let l = load(p.to_str().unwrap()).unwrap();
    assert_eq!(l.tok.vocab_size(), 256);
    assert!(ok(&format!(
        "generate --device cpu --model {} --prompt ab --max-new 3",
        p.display()
    ))
    .starts_with("ab"));
    // 100 tokens with no stored tokenizer cannot be decoded
    let cfg = Config { vocab: 100, ..cfg };
    let q = d.join("q.gguf");
    std::fs::write(&q, to_gguf(&Gpt2::init(cfg, 1), Quant::F32)).unwrap();
    assert!(load(q.to_str().unwrap())
        .err()
        .unwrap()
        .contains("no tokenizer"));
}

#[test]
fn mean_loss_needs_two_tokens_and_is_finite() {
    use ferrum_ref::gpt2::{Config, Gpt2};
    let m = Gpt2::init(
        Config {
            vocab: 20,
            ctx: 8,
            dim: 8,
            heads: 2,
            layers: 1,
            hidden: 16,
        },
        1,
    );
    let (l, n) = score::mean_loss(&m, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 4).unwrap();
    assert!(l.is_finite() && l > 0.0 && n > 0);
    assert!(score::mean_loss(&m, &[1], 4)
        .unwrap_err()
        .contains("at least 2"));
}

#[test]
fn export_gguf_changes_precision_and_keeps_the_tokenizer() {
    let d = dir("export");
    let (c, m) = (data(&d), d.join("m.gguf"));
    ok(&format!(
        "train {FAST} --epochs 8 --data {c} --out {} --quant f32",
        m.display()
    ));
    let (i8p, i4p) = (d.join("i8.gguf"), d.join("i4.gguf"));
    let r = ok(&format!(
        "export-gguf --model {} --out {} --quant int8",
        m.display(),
        i8p.display()
    ));
    assert!(r.starts_with("wrote ") && r.contains(" bytes"), "{r}");
    ok(&format!(
        "export-gguf --model {} --out {} --quant int4",
        m.display(),
        i4p.display()
    ));
    let size = |p: &std::path::Path| std::fs::metadata(p).unwrap().len();
    assert!(
        size(&i4p) < size(&i8p) && size(&i8p) < size(&m),
        "{} {} {}",
        size(&i4p),
        size(&i8p),
        size(&m)
    );
    for p in [&i8p, &i4p] {
        assert!(is_gpt2(p.to_str().unwrap()));
        assert!(ok(&format!(
            "generate --device cpu --model {} --prompt the --max-new 4",
            p.display()
        ))
        .starts_with("the"));
    }
    // the tokenizer survived: same vocabulary, and the int8 file scores close to the f32 one
    let loss = |p: &std::path::Path| {
        let e = ok(&format!("eval --model {} --data {c}", p.display()));
        e.split("loss: ")
            .nth(1)
            .unwrap()
            .split(' ')
            .next()
            .unwrap()
            .parse::<f64>()
            .unwrap()
    };
    assert_eq!(
        load(i8p.to_str().unwrap()).unwrap().tok.vocab_size(),
        load(m.to_str().unwrap()).unwrap().tok.vocab_size()
    );
    assert!((loss(&i8p) - loss(&m)).abs() < 0.1);
    // a Llama-style file converts too; errors are by name
    let l = d.join("l.gguf");
    ok(&format!("train --arch llama --device cpu --preset tiny --data {c} --out {} --steps 3 --batch 2 --context 8", l.display()));
    ok(&format!(
        "export-gguf --model {} --out {} --quant int8",
        l.display(),
        d.join("l8.gguf").display()
    ));
    assert!(err("export-gguf --model /nope --out x").contains("/nope"));
    assert!(err(&format!(
        "export-gguf --model {} --out x --quant q9",
        m.display()
    ))
    .contains("--quant"));
    assert!(err("export-gguf --out x").contains("--model is required"));
}

#[test]
fn tokenize_writes_the_same_file_as_ferrum_cpu() {
    let d = dir("tokenize_golden");
    let (c, t, o) = (d.join("c.txt"), d.join("tok.gguf"), d.join("o.tok.gguf"));
    std::fs::write(&c, FIX_CORPUS).unwrap();
    std::fs::write(&t, FIX_TOK).unwrap();
    let r = ok(&format!(
        "tokenize --data {} --tokenizer {} --out {}",
        c.display(),
        t.display(),
        o.display()
    ));
    assert!(
        r.contains("942 tokens, 31 documents, vocab 320 (2010 bytes in, 2332 bytes out)"),
        "{r}"
    );
    assert_eq!(std::fs::read(&o).unwrap(), FIX_TOKENS);
}
