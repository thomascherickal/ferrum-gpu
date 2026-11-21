use super::*;

#[test]
fn finetune_continues_from_a_gpt2_file_with_its_tokenizer() {
    let d = dir("ft");
    let (c, m, g) = (data(&d), d.join("m.gguf"), d.join("g.gguf"));
    ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {}",
        m.display()
    ));
    let t = ok(&format!(
        "finetune-gguf --device cpu --model {} --data {c} --out {} --epochs 2 --batch 4",
        m.display(),
        g.display()
    ));
    assert!(t.contains("steps: 14") && !t.contains("tokenizer:"), "{t}");
    assert!(is_gpt2(g.to_str().unwrap()));
    assert!(err(&format!(
        "finetune-gguf --device cpu --model {} --data {c} --out x",
        "/nope"
    ))
    .contains("/nope"));
}

#[test]
fn eval_reports_loss_and_writes_json() {
    let d = dir("eval");
    let (c, m, j) = (data(&d), d.join("m.gguf"), d.join("e.json"));
    ok(&format!(
        "train {FAST} --epochs 20 --data {c} --out {}",
        m.display()
    ));
    let e = ok(&format!(
        "eval --model {} --data {c} --ctx 16 --json {}",
        m.display(),
        j.display()
    ));
    assert!(e.contains("tokens: ") && e.contains("perplexity: "), "{e}");
    let json = std::fs::read_to_string(&j).unwrap();
    assert!(
        json.starts_with("{\"tokens\":") && json.contains("\"perplexity\":"),
        "{json}"
    );
    assert!(err(&format!(
        "eval --model {} --data {}",
        m.display(),
        d.join("tiny.txt").display()
    ))
    .contains("--data"));
    std::fs::write(d.join("one.txt"), "a").unwrap();
    assert!(err(&format!(
        "eval --model {} --data {}",
        m.display(),
        d.join("one.txt").display()
    ))
    .contains("at least 2"));
}

#[test]
fn unsupported_and_misplaced_options_are_refused_by_name() {
    let d = dir("refuse");
    let (c, o) = (data(&d), d.join("o.gguf").display().to_string());
    let t = |extra: &str| err(&format!("train --device cpu --data {c} --out {o} {extra}"));
    assert!(t("--preset tiny --dropout 1.5").contains("--dropout must be in [0, 1)"));
    assert!(t("--preset tiny --arch bert").contains("unknown '"));
    assert!(t("--preset tiny --arch llama --epochs 3").contains("--epochs needs --arch gpt2"));
    assert!(t("--preset tiny --arch llama --embed 8").contains("--embed needs --arch gpt2"));
    assert!(t("--preset huge").contains("unknown --preset"));
    assert!(t("--preset tiny --vocab 100").contains("tokenizer"));
    assert!(err(&format!("train --device cpu --data /nope --out {o}")).contains("--data"));
    let bin = d.join("bin.txt");
    std::fs::write(&bin, [0xffu8, 0xfe, 0x00]).unwrap();
    assert!(err(&format!(
        "train --device cpu --data {} --out {o}",
        bin.display()
    ))
    .contains("not valid UTF-8"));
}
