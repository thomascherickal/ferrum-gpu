use super::*;

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
