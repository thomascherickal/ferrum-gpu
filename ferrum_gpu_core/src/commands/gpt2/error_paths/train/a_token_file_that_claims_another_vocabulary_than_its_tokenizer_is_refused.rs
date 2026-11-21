use super::*;

#[test]
pub(super) fn a_token_file_that_claims_another_vocabulary_than_its_tokenizer_is_refused() {
    let (d, corpus) = scratch("tfvocab");
    let (_, tok) = trained(&d, &corpus, 300, "a");
    let tok_bytes = std::fs::read(&tok).unwrap();
    let real = ferrum_model::tokenizer_gguf::from_gguf_bytes(&tok_bytes).unwrap();
    // a token file whose hash is the tokenizer's but whose vocabulary is five larger
    let tf = d.join("lying.tok.gguf");
    let meta = TokenFileMeta {
        vocab_size: real.vocab_size() + 5,
        eos_id: None,
        tokenizer_sha256: sha256_hex(&tok_bytes),
    };
    write(tf.to_str().unwrap(), &meta, |sink| {
        sink.push(&(1..=20).collect::<Vec<u32>>())
    })
    .unwrap();
    let e = err(&format!(
        "train {FAST} --data {} --tokenizer {} --out {}",
        tf.display(),
        tok.display(),
        d.join("o.gguf").display()
    ));
    assert!(e.contains("was made for a vocabulary of"), "{e}");
}

#[test]
pub(super) fn a_token_file_needs_its_tokenizer() {
    let (d, corpus) = scratch("tfneeds");
    let (_, tok) = trained(&d, &corpus, 300, "a");
    let tf = d.join("c.tokens.gguf");
    ok(&format!(
        "tokenize --data {} --tokenizer {} --out {}",
        corpus.display(),
        tok.display(),
        tf.display()
    ));
    let e = err(&format!(
        "train {FAST} --data {} --out {}",
        tf.display(),
        d.join("o.gguf").display()
    ));
    assert!(e.contains("needs --tokenizer"), "{e}");
}

#[test]
pub(super) fn resuming_with_a_tokenizer_of_the_same_size_but_other_merges_is_refused() {
    let (d, corpus_a) = scratch("resume");
    // a second corpus of different words: the same --vocab, different merges
    let corpus_b = d.join("b.txt");
    std::fs::write(&corpus_b, "zebra quartz jumped over fixed wax. ".repeat(40)).unwrap();
    let ck = d.join("ck.gguf");
    ok(&format!(
        "train {FAST} --epochs 2 --vocab 262 --data {} --out {} --tokenizer {} --checkpoint {} --checkpoint-every 1",
        corpus_a.display(),
        d.join("a.gguf").display(),
        d.join("a.tok.gguf").display(),
        ck.display()
    ));
    let e = err(&format!(
        "train {FAST} --epochs 4 --vocab 262 --data {} --out {} --tokenizer {} --resume {}",
        corpus_b.display(),
        d.join("b.gguf").display(),
        d.join("b.tok.gguf").display(),
        ck.display()
    ));
    assert!(
        e.contains("cannot resume") && e.contains("different tokenizer"),
        "{e}"
    );
}
