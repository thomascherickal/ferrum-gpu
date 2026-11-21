use super::*;

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
