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
