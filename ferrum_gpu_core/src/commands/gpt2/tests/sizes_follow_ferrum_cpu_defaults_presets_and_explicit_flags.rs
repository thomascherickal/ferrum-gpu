use super::*;

#[test]
fn sizes_follow_ferrum_cpu_defaults_presets_and_explicit_flags() {
    let table = [&SIZE_FLAGS[..]].concat();
    let parse = |s: &str| Args::parse(&table, &words(s)).unwrap();
    let d = Sizes::from_flags(&parse("")).unwrap();
    assert_eq!(d, Sizes::DEFAULT);
    let small = Sizes::from_flags(&parse("--preset small")).unwrap();
    assert_eq!(
        (
            small.ctx,
            small.dim,
            small.layers,
            small.hidden,
            small.vocab
        ),
        (128, 128, 4, 512, 2048)
    );
    assert_eq!(small.lr, 3e-3);
    for order in [
        "--preset tiny --embed 32 --blocks 3",
        "--embed 32 --blocks 3 --preset tiny",
    ] {
        let s = Sizes::from_flags(&parse(order)).unwrap();
        assert_eq!(
            (s.dim, s.layers, s.ctx, s.hidden, s.vocab),
            (32, 3, 64, 256, 512),
            "{order}"
        );
    }
    assert!(Sizes::from_flags(&parse("--preset huge"))
        .unwrap_err()
        .contains("valid: tiny, small, medium"));
    assert!(Sizes::from_flags(&parse("--embed x"))
        .unwrap_err()
        .contains("--embed"));
}

#[test]
fn vocab_zero_trains_a_character_level_model_that_generates_and_reloads() {
    let d = dir("charlevel");
    let c = data(&d);
    let (o, tok) = (d.join("c.gguf"), d.join("c.tok.gguf"));
    let t = ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --vocab 0 --tokenizer {}",
        o.display(),
        tok.display()
    ));
    assert!(t.contains("character-level (bytes) 256 tokens"), "{t}");
    assert!(t.contains("vocab 256"), "{t}");
    // the model reloads and generates from the saved tokenizer, one token per character
    let g = ok(&format!(
        "generate --device cpu --model {} --prompt ab --max-new 5",
        o.display()
    ));
    assert!(g.starts_with("ab"), "{g}");
    // a later run reuses the tokenizer file (no --vocab needed)
    let again = ok(&format!(
        "train {FAST} --epochs 1 --data {c} --out {} --tokenizer {}",
        d.join("c2.gguf").display(),
        tok.display()
    ));
    assert!(again.contains("256 tokens"), "{again}");
}

#[test]
fn tie_trains_a_shared_embedding_and_head_that_the_file_stores_once() {
    let d = dir("tie");
    let c = data(&d);
    let (plain, tied) = (d.join("plain.gguf"), d.join("tied.gguf"));
    ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --vocab 300",
        plain.display()
    ));
    let t = ok(&format!(
        "train {FAST} --epochs 3 --data {c} --out {} --vocab 300 --tie",
        tied.display()
    ));
    assert!(!t.contains("training on the CPU engine"), "{t}");
    let has_head = |p: &std::path::Path| {
        ferrum_gguf::Gguf::parse(std::fs::read(p).unwrap())
            .unwrap()
            .tensor("output.weight")
            .is_some()
    };
    assert!(has_head(&plain), "an ordinary model stores its own head");
    assert!(!has_head(&tied), "a tied model stores the embedding only");
    assert!(
        std::fs::metadata(&tied).unwrap().len() < std::fs::metadata(&plain).unwrap().len(),
        "so the file is smaller"
    );
    // it loads and generates like any other file
    let g = ok(&format!(
        "generate --device cpu --model {} --prompt the --max-new 6",
        tied.display()
    ));
    assert!(g.starts_with("the"), "{g}");
    // device auto falls back to the CPU engine for --tie rather than failing
    ok(&format!(
        "train --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 4 --epochs 1 --data {c} --out {} --vocab 300 --tie",
        d.join("auto.gguf").display()
    ));
}

#[test]
fn quant_accepts_both_spellings() {
    let t = [crate::commands::flags::val("--quant")];
    let q = |s: &str| quant(&Args::parse(&t, &words(s)).unwrap(), "int8");
    assert_eq!(q("--quant int4").unwrap(), Quant::Q4_0);
    assert_eq!(q("--quant q4_0").unwrap(), Quant::Q4_0);
    assert_eq!(q("--quant int8").unwrap(), Quant::Q8_0);
    assert_eq!(q("--quant f32").unwrap(), Quant::F32);
    assert_eq!(q("").unwrap(), Quant::Q8_0);
    for (s, want) in [
        ("--quant f16", Quant::F16),
        ("--quant fp16", Quant::F16),
        ("--quant fp4", Quant::Fp4),
        ("--quant mxfp4", Quant::Fp4),
        ("--quant q2", Quant::Q2K),
        ("--quant q2_k", Quant::Q2K),
    ] {
        assert_eq!(q(s).unwrap(), want, "{s}");
    }
    let e = q("--quant fp9").unwrap_err();
    assert!(
        e.contains("unknown 'fp9'") && e.contains(QUANT_NAMES),
        "{e}"
    );
}
