use super::*;

#[test]
fn the_files_of_a_tokenizer_without_end_of_text_are_joined_with_no_separator_token() {
    let d = dir("tok_noeos");
    let tok = d.join("byte.tok.gguf");
    let byte = ferrum_tok::ByteBpeTokenizer::byte_level();
    assert!(byte.eos_id().is_none());
    std::fs::write(&tok, ferrum_model::tokenizer_gguf::to_gguf_bytes(&byte)).unwrap();
    let docs = d.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("a.txt"), "ab").unwrap();
    std::fs::write(docs.join("b.txt"), "cd").unwrap();
    let o = d.join("o.tok.gguf");
    ok(&format!(
        "tokenize --data {} --tokenizer {} --out {}",
        docs.display(),
        tok.display(),
        o.display()
    ));
    let f = ferrum_model::token_file::TokenFile::open(o.to_str().unwrap()).unwrap();
    assert_eq!(f.read(0, 4).unwrap(), vec![97, 98, 99, 100]);
}

#[test]
fn a_streamed_reply_that_ends_inside_a_character_still_emits_its_tail() {
    use ferrum_model::sampling::Params;
    // byte 0xE2 starts a three-byte character: decoded alone it is U+FFFD, which streaming holds back
    // in case the next token completes it; when the run ends the held text is emitted
    let tok = ferrum_tok::ByteBpeTokenizer::byte_level();
    let mut feed = |_: usize| -> Result<Vec<f32>, String> {
        let mut l = vec![0.0f32; 256];
        l[0xE2] = 9.0;
        Ok(l)
    };
    let o = gen::Opts {
        max_new: 2,
        params: Params::default(),
        seed: 1,
        stops: vec![],
        stream: true,
        stop_ids: vec![],
    };
    let mut got = String::new();
    let mut emit = |s: &str| got.push_str(s);
    let (text, n) = gen::run(
        &mut feed,
        &|ids| tok.decode(ids),
        64,
        &[1],
        &o,
        Some(&mut emit),
    )
    .unwrap();
    assert_eq!(n, 2);
    assert_eq!(got, text, "everything produced is emitted by the end");
    assert!(text.contains('\u{fffd}'));
}
