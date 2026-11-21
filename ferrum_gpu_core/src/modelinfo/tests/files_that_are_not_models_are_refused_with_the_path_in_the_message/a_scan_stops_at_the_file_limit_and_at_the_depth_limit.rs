use super::*;

#[test]
fn a_scan_stops_at_the_file_limit_and_at_the_depth_limit() {
    let d = tmp();
    for i in 0..6 {
        write(&d, &format!("m{i}.gguf"), b"x");
    }
    let (files, truncated) = scan_with(&d, 4).unwrap();
    assert_eq!((files.len(), truncated), (4, true));
    let (files, truncated) = scan_with(&d, 6).unwrap();
    assert_eq!((files.len(), truncated), (6, false));
    let mut deep = String::new();
    for i in 0..=MAX_DEPTH + 1 {
        deep.push_str(&format!("l{i}/"));
    }
    write(&d, &format!("{deep}too_deep.gguf"), b"x");
    let (files, _) = scan(&d).unwrap();
    assert!(!files.iter().any(|f| f.path.ends_with("too_deep.gguf")));
}

#[test]
fn scanning_a_missing_folder_or_a_file_is_an_error_naming_it() {
    let d = tmp();
    assert!(scan(&d.join("nope")).unwrap_err().contains("nope"));
    let f = write(&d, "f.gguf", b"x");
    assert!(scan(&f).is_err());
    assert_eq!(scan(&d).unwrap().0.len(), 1);
}

#[test]
fn the_json_carries_the_tokenizer_of_a_model_that_has_one() {
    let d = tmp();
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("qwen2".into()));
    b.meta("tokenizer.ggml.model", Value::Str("gpt2".into()));
    b.meta(
        "tokenizer.ggml.tokens",
        Value::Arr(vec![
            Value::Str("a".into()),
            Value::Str("<|im_start|>".into()),
        ]),
    );
    b.meta("tokenizer.chat_template", Value::Str("<|im_start|>".into()));
    b.tensor_f32("token_embd.weight", &[2, 2], &[0.5; 4]);
    let j = to_json(&inspect(&write(&d, "t.gguf", &b.into_bytes())).unwrap());
    let t = j.get("tokenizer").unwrap();
    assert_eq!(t.get("model").and_then(Json::as_str), Some("gpt2"));
    assert_eq!(t.get("tokens").and_then(Json::as_f64), Some(2.0));
    assert_eq!(t.get("chat_format").and_then(Json::as_str), Some("ChatML"));
    // a model without a vocabulary has null
    let plain = to_json(&inspect(&write(&d, "p.gguf", &model_bytes())).unwrap());
    assert_eq!(plain.get("tokenizer"), Some(&Json::Null));
}

#[cfg(unix)]
#[test]
fn a_scan_skips_an_unreadable_folder_and_a_dangling_link() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let d = tmp();
    write(&d, "m.gguf", &model_bytes());
    let locked = d.join("locked");
    std::fs::create_dir(&locked).unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    symlink(d.join("missing"), d.join("dangling.gguf")).unwrap();
    let r = scan(&d);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (files, truncated) = r.unwrap();
    assert_eq!((files.len(), truncated), (1, false));
}
