use super::*;

#[test]
fn files_that_are_not_models_are_refused_with_the_path_in_the_message() {
    let d = tmp();
    let text = write(&d, "text.gguf", b"this is not a gguf file at all");
    assert!(inspect(&text).unwrap_err().contains("magic"));
    let empty = write(&d, "empty.gguf", b"");
    assert!(inspect(&empty).is_err());
    let good = model_bytes();
    for cut in [4, 12, 40, good.len() - 70] {
        let p = write(&d, &format!("cut{cut}.gguf"), &good[..cut]);
        assert!(inspect(&p).is_err(), "a file cut at {cut} bytes is refused");
    }
    assert!(inspect(&d).unwrap_err().contains("not a file"));
    assert!(inspect(&d.join("nope.gguf"))
        .unwrap_err()
        .contains("nope.gguf"));
    // a header claiming a billion entries is refused, not allocated
    let mut hostile = good[..8].to_vec();
    hostile.extend_from_slice(&(u64::MAX).to_le_bytes());
    hostile.extend_from_slice(&(u64::MAX).to_le_bytes());
    let p = write(&d, "hostile.gguf", &hostile);
    assert!(inspect(&p).is_err());
}

#[test]
fn the_json_has_every_field_the_page_reads() {
    let d = tmp();
    let i = inspect(&write(&d, "m.gguf", &model_bytes())).unwrap();
    let j = to_json(&i);
    for k in [
        "file_size",
        "version",
        "architecture",
        "name",
        "n_tensors",
        "n_kv",
        "parameters",
        "tensor_bytes",
        "kinds",
        "metadata",
        "tensors",
        "truncated",
        "tokenizer",
    ] {
        assert!(j.get(k).is_some(), "{k}");
    }
    assert_eq!(j.get("architecture").and_then(Json::as_str), Some("gpt2"));
    let Some(Json::Arr(ts)) = j.get("tensors") else {
        panic!()
    };
    assert_eq!(ts[1].get("kind").and_then(Json::as_str), Some("Q8_0"));
    assert_eq!(ts[1].get("bytes").and_then(Json::as_f64), Some(68.0));
    let none = to_json(&Info {
        architecture: None,
        name: None,
        ..i
    });
    assert_eq!(none.get("name"), Some(&Json::Null));
}

#[test]
fn scanning_finds_gguf_files_in_subfolders_sorted_and_skips_everything_else() {
    let d = tmp();
    let m = model_bytes();
    write(&d, "b.gguf", &m);
    write(&d, "A.GGUF", &m);
    write(&d, "notes.txt", b"x");
    write(&d, "sub/deep.gguf", &m);
    write(&d, ".hidden/secret.gguf", &m);
    let (files, truncated) = scan(&d).unwrap();
    let names: Vec<_> = files
        .iter()
        .map(|f| f.path.file_name().unwrap().to_str().unwrap().to_string())
        .collect();
    assert_eq!(names, ["A.GGUF", "b.gguf", "deep.gguf"]);
    assert!(!truncated);
    assert!(files
        .iter()
        .all(|f| f.size == m.len() as u64 && f.modified.is_some()));
}

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
