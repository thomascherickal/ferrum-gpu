use super::*;

#[test]
fn a_llama_style_file_serves_all_five_routes() {
    let dir = work("llama");
    let model = dir.join("m.gguf");
    std::fs::write(
        &model,
        to_gguf(&fresh_model("tiny", 1).unwrap(), Quant::F32),
    )
    .unwrap();
    let s = spawn(model.to_str().unwrap(), &[], &[]);
    let a = &s.addr;
    assert!(http(a, "GET", "/health", "", "").contains(r#""status":"ok""#));
    let info = http(a, "GET", "/v1/info", "", "");
    assert!(
        info.contains(r#""kind":"cpu""#)
            && info.contains(r#""vocab":256"#)
            && info.contains(r#""arch":"llama""#),
        "{info}"
    );
    let g = http(
        a,
        "POST",
        "/v1/generate",
        "",
        r#"{"requests":[{"prompt":"hi","max_tokens":5}]}"#,
    );
    assert!(g.contains("200") && g.contains(r#""n_tokens":5"#), "{g}");
    let l = http(
        a,
        "POST",
        "/v1/loglikelihood",
        "",
        r#"{"requests":[{"context":"hello","continuation":" world"}]}"#,
    );
    assert!(
        l.contains("200") && l.contains("logprob") && l.contains(r#""n_tokens":6"#),
        "{l}"
    );
    let r = http(
        a,
        "POST",
        "/v1/loglikelihood_rolling",
        "",
        r#"{"texts":["hello world"]}"#,
    );
    assert!(r.contains("200") && r.contains(r#""n_tokens":11"#), "{r}");
    assert!(http(a, "POST", "/v1/generate", "", "oops").contains("400"));
    std::fs::remove_dir_all(dir).ok();
}
