//! `generate` and `eval`: a model whose tokenizer does not fit it, and an output that closes.

use super::{err, ok, run_failing, scratch, trained};
use ferrum_model::gpt2_gguf::{to_gguf_with, Quant};
use ferrum_ref::gpt2::{Config, Gpt2};

#[test]
fn a_tokenizer_with_more_tokens_than_the_model_is_refused() {
    let (d, corpus) = scratch("bigtok");
    let (_, tok) = trained(&d, &corpus, 300, "a");
    let t = ferrum_model::tokenizer_gguf::from_gguf_bytes(&std::fs::read(&tok).unwrap()).unwrap();
    // a model ten tokens smaller than the tokenizer it carries
    let (big, small) = (t.vocab_size(), t.vocab_size() - 10);
    let cfg = Config {
        vocab: small,
        ctx: 16,
        dim: 16,
        heads: 2,
        layers: 1,
        hidden: 32,
    };
    let bad = d.join("bad.gguf");
    std::fs::write(
        &bad,
        to_gguf_with(&Gpt2::init(cfg, 1), Quant::F32, Some(&t)),
    )
    .unwrap();
    let e = err(&format!(
        "generate --device cpu --model {} --prompt the",
        bad.display()
    ));
    assert!(
        e.contains(&format!(
            "the tokenizer has {big} tokens but the model {small}"
        )),
        "{e}"
    );
    let e = err(&format!(
        "eval --model {} --data {}",
        bad.display(),
        corpus.display()
    ));
    assert!(
        e.contains(&format!("the tokenizer has {big} tokens")),
        "{e}"
    );
}

#[test]
fn a_model_with_no_tokenizer_and_no_byte_vocabulary_is_refused() {
    let (d, _) = scratch("notok");
    let cfg = Config {
        vocab: 300,
        ctx: 16,
        dim: 16,
        heads: 2,
        layers: 1,
        hidden: 32,
    };
    let bare = d.join("bare.gguf");
    std::fs::write(&bare, to_gguf_with(&Gpt2::init(cfg, 1), Quant::F32, None)).unwrap();
    let e = err(&format!(
        "generate --device cpu --model {} --prompt x",
        bare.display()
    ));
    assert!(e.contains("has no tokenizer for its 300 tokens"), "{e}");
}

#[test]
fn a_closed_output_is_an_error_not_a_panic() {
    let (d, corpus) = scratch("closed");
    let (m, _) = trained(&d, &corpus, 300, "m");
    let cmd = format!(
        "generate --device cpu --model {} --prompt the --max-new 4",
        m.display()
    );
    // nothing can be written at all
    let e = run_failing(&cmd, 0).unwrap_err();
    assert!(e.contains("output closed"), "{e}");
    // the prompt goes out, then the streamed text cannot
    let streamed = format!("{cmd} --stream");
    let e = run_failing(&streamed, "the".len()).unwrap_err();
    assert!(e.contains("output closed"), "{e}");
}

#[test]
fn run_gguf_and_generate_agree_on_the_text() {
    let (d, corpus) = scratch("agree");
    let (m, _) = trained(&d, &corpus, 300, "m");
    let g = ok(&format!(
        "generate --device cpu --model {} --prompt the --max-new 6",
        m.display()
    ));
    let r = ok(&format!(
        "run-gguf --device cpu --model {} --prompt the --max-new 6",
        m.display()
    ));
    // run-gguf prints the device and shape first, then the same text
    assert!(r.trim_end().ends_with(g.trim_end()), "{r}\n---\n{g}");
}
