use super::*;

#[test]
fn a_chat_model_is_wrapped_in_its_template_and_prints_only_the_reply() {
    let (m, _) = model("chat", &TOKS, true);
    let p = m.display();
    let chat = ok(&format!(
        "generate --model {p} --prompt ab --max-new 4 --device cpu"
    ));
    assert!(
        !chat.contains("<|im_start|>") && !chat.starts_with("ab"),
        "{chat:?}"
    );
    // raw: no template, the prompt is echoed
    let raw = ok(&format!(
        "generate --model {p} --prompt abab --max-new 4 --device cpu --raw"
    ));
    assert!(raw.starts_with("abab"), "{raw:?}");
    // the same seed gives the same text
    assert_eq!(
        chat,
        ok(&format!(
            "generate --model {p} --prompt ab --max-new 4 --device cpu"
        ))
    );
    // streaming prints the same characters, prompt first in raw mode
    let s = ok(&format!(
        "generate --model {p} --prompt abab --max-new 4 --device cpu --raw --stream"
    ));
    assert_eq!(s, raw);
    // run-gguf adds the device and shape lines
    let v = ok(&format!(
        "run-gguf --model {p} --prompt ab --max-new 2 --device cpu"
    ));
    assert!(
        v.contains("device: CPU") && v.contains("model: vocab 12"),
        "{v}"
    );
}

#[test]
fn ids_bypass_the_template_and_bad_ids_are_named() {
    let (m, _) = model("ids", &TOKS, true);
    let p = m.display();
    let a = [
        "generate",
        "--model",
        &p.to_string(),
        "--ids",
        "0 1",
        "--max-new",
        "3",
        "--device",
        "cpu",
    ]
    .map(String::from);
    let mut out = Vec::new();
    cli(&a, &mut out).unwrap();
    assert!(!out.is_empty());
    let bad = [
        "generate",
        "--model",
        &p.to_string(),
        "--ids",
        "0 x",
        "--device",
        "cpu",
    ]
    .map(String::from);
    assert!(cli(&bad, &mut Vec::new())
        .unwrap_err()
        .contains("'x' is not an id"));
    let empty = [
        "generate",
        "--model",
        &p.to_string(),
        "--ids",
        "",
        "--device",
        "cpu",
    ]
    .map(String::from);
    assert!(cli(&empty, &mut Vec::new())
        .unwrap_err()
        .contains("prompt is empty"));
    assert!(err(&format!("generate --model {p} --device cpu")).contains("--prompt"));
}
