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
