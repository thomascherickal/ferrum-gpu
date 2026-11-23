//! Inputs that exist but cannot be used: a directory where a file belongs, a file that is not a
//! GGUF, a directory that cannot be listed. Each failure names the path it came from.

use super::{err, scratch, trained, FAST};
use ferrum_model::sha256::sha256_hex;
use ferrum_model::token_file::{write, TokenFileMeta};

#[test]
fn tokenize_names_an_unlistable_directory_and_a_tokenizer_that_is_not_one() {
    let (d, c) = scratch("tok_in");
    let (_, tok) = trained(&d, &c, 300, "m");
    let out = d.join("o.gguf");
    // /proc/1/fd is a directory an ordinary user cannot list
    let e = err(&format!(
        "tokenize --data /proc/1/fd --tokenizer {} --out {}",
        tok.display(),
        out.display()
    ));
    assert!(e.starts_with("/proc/1/fd: "), "{e}");
    let e = err(&format!(
        "tokenize --data {} --tokenizer {} --out {}",
        c.display(),
        c.display(),
        out.display()
    ));
    assert_eq!(e, format!("{}: not a GGUF tokenizer file", c.display()));
}

#[test]
fn train_names_a_resume_file_that_is_a_directory_or_not_a_checkpoint() {
    let (d, c) = scratch("resume_in");
    let run = |resume: &std::path::Path| {
        err(&format!(
            "train {FAST} --vocab 300 --data {} --out {} --resume {}",
            c.display(),
            d.join("o.gguf").display(),
            resume.display()
        ))
    };
    assert!(run(&d).starts_with(&format!("{}: ", d.display())));
    let e = run(&c);
    assert!(
        e.starts_with(&format!("{}: ", c.display())) && e.contains("magic"),
        "{e}"
    );
}

#[test]
fn train_names_a_tokenizer_file_that_is_a_directory_or_not_a_tokenizer() {
    let (d, c) = scratch("tok_file_in");
    let run = |tok: &std::path::Path| {
        err(&format!(
            "train {FAST} --vocab 300 --data {} --out {} --tokenizer {}",
            c.display(),
            d.join("o.gguf").display(),
            tok.display()
        ))
    };
    assert!(run(&d).starts_with(&format!("{}: ", d.display())));
    let e = run(&c);
    assert!(
        e.starts_with(&format!("{}: not a GGUF", c.display())),
        "{e}"
    );
}

#[test]
fn a_token_file_names_a_tokenizer_that_is_a_directory_or_a_hash_match_that_is_not_gguf() {
    let (d, c) = scratch("tok_file_tok");
    let (_, tok) = trained(&d, &c, 300, "m");
    let make = |tag: &str, sha: String| {
        let tf = d.join(format!("{tag}.tokens.gguf"));
        let meta = TokenFileMeta {
            vocab_size: 300,
            eos_id: None,
            tokenizer_sha256: sha,
        };
        write(tf.to_str().unwrap(), &meta, |s| {
            s.push(&(1..=20).collect::<Vec<u32>>())
        })
        .unwrap();
        tf
    };
    let run = |tf: &std::path::Path, tokenizer: &std::path::Path| {
        err(&format!(
            "train {FAST} --data {} --out {} --tokenizer {}",
            tf.display(),
            d.join("o.gguf").display(),
            tokenizer.display()
        ))
    };
    let e = run(&make("dir", sha256_hex(&std::fs::read(&tok).unwrap())), &d);
    assert!(e.starts_with(&format!("{}: ", d.display())), "{e}");
    // the file's hash is the token file's, but it is not a tokenizer
    let e = run(&make("text", sha256_hex(&std::fs::read(&c).unwrap())), &c);
    assert!(
        e.starts_with(&format!("{}: not a GGUF", c.display())),
        "{e}"
    );
}
