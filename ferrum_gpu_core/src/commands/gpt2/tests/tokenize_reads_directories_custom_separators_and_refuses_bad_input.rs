use super::*;

#[test]
fn tokenize_reads_directories_custom_separators_and_refuses_bad_input() {
    let d = dir("tokenize_more");
    let (c, tok) = (data(&d), d.join("tok.gguf"));
    ok(&format!(
        "train {FAST} --epochs 1 --data {c} --tokenizer {} --out {}",
        tok.display(),
        d.join("m.gguf").display()
    ));
    let docs = d.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("b.txt"), "the mat. ").unwrap();
    std::fs::write(docs.join("a.txt"), "the cat. ").unwrap();
    let o = d.join("d.tok.gguf");
    let r = ok(&format!(
        "tokenize --data {} --tokenizer {} --out {}",
        docs.display(),
        tok.display(),
        o.display()
    ));
    assert!(r.contains("2 documents"), "{r}"); // one end-of-text id between the two files
    let f = ferrum_model::token_file::TokenFile::open(o.to_str().unwrap()).unwrap();
    let t = load(d.join("m.gguf").to_str().unwrap()).unwrap().tok;
    let want = [
        t.encode_docs("the cat. "),
        vec![t.eos_id().unwrap()],
        t.encode_docs("the mat. "),
    ]
    .concat();
    assert_eq!(f.read(0, want.len()).unwrap(), want, "files in path order");
    // a custom separator becomes the end-of-text id
    let s = d.join("s.txt");
    std::fs::write(&s, "one||two||three").unwrap();
    let r = ok(&format!(
        "tokenize --data {} --tokenizer {} --out {} --doc-sep ||",
        s.display(),
        tok.display(),
        o.display()
    ));
    assert!(r.contains("3 documents"), "{r}");
    let empty = d.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    assert!(err(&format!(
        "tokenize --data {} --tokenizer {} --out {}",
        empty.display(),
        tok.display(),
        o.display()
    ))
    .contains("holds no files"));
    let empty_sep: Vec<String> = [
        "tokenize",
        "--data",
        s.to_str().unwrap(),
        "--tokenizer",
        tok.to_str().unwrap(),
        "--out",
        o.to_str().unwrap(),
        "--doc-sep",
        "",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    assert!(run(&empty_sep, &mut Vec::new())
        .unwrap_err()
        .contains("--doc-sep must not be empty"));
    let bin = d.join("bin");
    std::fs::write(&bin, [0xffu8, 0xfe]).unwrap();
    assert!(err(&format!(
        "tokenize --data {} --tokenizer {} --out {}",
        bin.display(),
        tok.display(),
        o.display()
    ))
    .contains("not valid UTF-8"));
    assert!(err(&format!(
        "tokenize --data {} --tokenizer /nope --out {}",
        s.display(),
        o.display()
    ))
    .contains("/nope"));
    assert!(err(&format!(
        "tokenize --data {} --tokenizer {} --out {}",
        d.join("nofile").display(),
        tok.display(),
        o.display()
    ))
    .contains("nofile"));
    assert!(err("tokenize --data x").contains("--tokenizer is required"));
}
