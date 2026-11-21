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
