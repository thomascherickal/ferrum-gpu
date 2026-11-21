use super::*;

#[test]
fn a_gpt2_file_serves_with_its_bpe_tokenizer() {
    let dir = work("gpt2");
    let (data, model) = (dir.join("c.txt"), dir.join("m.gguf"));
    std::fs::write(&data, "the cat sat on the mat. ".repeat(60)).unwrap();
    let cmd: Vec<String> = format!(
        "train --device cpu --context 16 --embed 16 --heads 2 --blocks 1 --hidden 32 --batch 8 --epochs 10 --vocab 300 --data {} --out {}",
        data.display(),
        model.display()
    )
    .split_whitespace()
    .map(String::from)
    .collect();
    ferrum_gpu_core::commands::run(&cmd, &mut Vec::new()).unwrap();
    let s = spawn(model.to_str().unwrap(), &[], &[]);
    let a = &s.addr;
    let info = http(a, "GET", "/v1/info", "", "");
    assert!(
        info.contains(r#""arch":"gpt2""#) && info.contains(r#""vocab":269"#),
        "{info}"
    );
    let g = http(
        a,
        "POST",
        "/v1/generate",
        "",
        r#"{"requests":[{"prompt":"the c","max_tokens":6,"stop":["zzz"]}]}"#,
    );
    assert!(g.contains("200") && g.contains(r#""text":""#), "{g}");
    let l = http(
        a,
        "POST",
        "/v1/loglikelihood",
        "",
        r#"{"requests":[{"context":"the cat sat","continuation":" on the mat."}]}"#,
    );
    let n = l
        .split(r#""n_tokens":"#)
        .nth(1)
        .and_then(|x| x.split([',', '}']).next())
        .and_then(|x| x.parse::<usize>().ok());
    assert!(
        l.contains("200") && n.is_some_and(|n| n > 0 && n < 12),
        "BPE should use fewer tokens than characters: {l}"
    );
    assert!(http(
        a,
        "POST",
        "/v1/loglikelihood_rolling",
        "",
        r#"{"texts":["the cat sat on the mat. the cat sat"]}"#
    )
    .contains("200"));
    assert!(http(
        a,
        "POST",
        "/v1/generate",
        "",
        r#"{"requests":[{"prompt":"x","chat":true}]}"#
    )
    .contains("400"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn the_token_gate_works_through_the_binary_and_the_environment() {
    let dir = work("auth");
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    for (extra, env) in [
        (vec!["--token", "s3cret"], vec![]),
        (vec![], vec![("FERRUM_TOKEN", "s3cret")]),
    ] {
        let s = spawn(model.to_str().unwrap(), &extra, &env);
        let a = &s.addr;
        assert!(http(a, "GET", "/health", "", "").contains("200"));
        assert!(http(a, "GET", "/v1/info", "", "").contains("401"));
        assert!(http(a, "GET", "/v1/info", "Authorization: Bearer nope\r\n", "").contains("401"));
        assert!(http(a, "GET", "/v1/info", "Authorization: Bearer s3cret\r\n", "").contains("200"));
    }
    std::fs::remove_dir_all(dir).ok();
}
