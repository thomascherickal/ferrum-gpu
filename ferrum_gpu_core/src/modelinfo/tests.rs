use super::scan::{scan, scan_with, MAX_DEPTH};
use super::*;
use ferrum_gguf::GgufBuilder;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

static N: AtomicU32 = AtomicU32::new(0);

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ferrum_modelinfo_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A small GGUF: a string, a number, a long string, an array, an f32 tensor and a Q8_0 tensor.
fn model_bytes() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    b.meta("general.name", Value::Str("tiny test".into()));
    b.meta("gpt2.context_length", Value::U32(64));
    b.meta("note", Value::Str("x".repeat(500)));
    b.meta("tokens", Value::Arr((0..20).map(Value::U32).collect()));
    b.tensor_f32("token_embd.weight", &[4, 3], &[0.5; 12]);
    b.tensor_raw("blk.0.ffn.weight", &[32, 2], 8, vec![0u8; 2 * 34]);
    b.into_bytes()
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, bytes).unwrap();
    p
}

#[test]
fn a_model_is_described_by_its_header_metadata_and_tensors() {
    let d = tmp();
    let p = write(&d, "m.gguf", &model_bytes());
    let i = inspect(&p).unwrap();
    assert_eq!(i.version, 3);
    assert_eq!(i.architecture.as_deref(), Some("gpt2"));
    assert_eq!(i.name.as_deref(), Some("tiny test"));
    assert_eq!((i.n_tensors, i.n_kv), (2, 5));
    assert_eq!(i.parameters, 12 + 64);
    assert_eq!(i.tensor_bytes, 12 * 4 + 2 * 34);
    assert_eq!(i.file_size, std::fs::metadata(&p).unwrap().len());
    assert!(!i.truncated);
    assert_eq!(i.tensors[0].name, "token_embd.weight");
    assert_eq!(i.tensors[0].dims, [4, 3]);
    assert_eq!(
        (i.tensors[0].kind.as_str(), i.tensors[0].bytes),
        ("F32", Some(48))
    );
    assert_eq!(
        (i.tensors[1].kind.as_str(), i.tensors[1].bytes),
        ("Q8_0", Some(68))
    );
    // the kinds are totalled, biggest first
    assert_eq!(
        i.kinds,
        vec![
            KindTotal {
                kind: "Q8_0".into(),
                tensors: 1,
                bytes: 68
            },
            KindTotal {
                kind: "F32".into(),
                tensors: 1,
                bytes: 48
            },
        ]
    );
}

#[test]
fn metadata_values_are_shown_cut_and_summarised() {
    let d = tmp();
    let i = inspect(&write(&d, "m.gguf", &model_bytes())).unwrap();
    let get = |k: &str| i.metadata.iter().find(|(n, _)| n == k).unwrap().1.clone();
    assert_eq!(get("general.architecture"), "\"gpt2\"");
    assert_eq!(get("gpt2.context_length"), "64");
    let note = get("note");
    assert!(
        note.ends_with('…') && note.chars().count() <= MAX_VALUE_CHARS + 1,
        "{note}"
    );
    let toks = get("tokens");
    assert!(toks.starts_with("[20 items] 0, 1, 2, 3, 4, 5"), "{toks}");
    // every value type has a display
    for (v, want) in [
        (Value::U8(1), "1"),
        (Value::I8(-2), "-2"),
        (Value::U16(3), "3"),
        (Value::I16(-4), "-4"),
        (Value::U32(5), "5"),
        (Value::I32(-6), "-6"),
        (Value::U64(7), "7"),
        (Value::I64(-8), "-8"),
        (Value::F32(1.5), "1.5"),
        (Value::F64(2.5), "2.5"),
        (Value::Bool(true), "true"),
    ] {
        assert_eq!(show(&v), want);
    }
}

#[test]
fn a_model_with_no_name_or_architecture_still_inspects() {
    let d = tmp();
    let mut b = GgufBuilder::new();
    b.tensor_f32("w", &[2], &[1.0, 2.0]);
    let i = inspect(&write(&d, "bare.gguf", &b.into_bytes())).unwrap();
    assert_eq!((i.architecture, i.name, i.n_kv), (None, None, 0));
    // a non-string architecture is not an architecture
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::U32(1));
    assert_eq!(
        inspect(&write(&d, "odd.gguf", &b.into_bytes()))
            .unwrap()
            .architecture,
        None
    );
}

#[test]
fn only_the_start_of_a_big_file_is_read_and_a_header_over_the_limit_is_said_to_be_so() {
    let d = tmp();
    // 5 MB of "tensor data" behind a header of a few hundred bytes: a 4 KiB prefix is enough
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    b.tensor_raw("big", &[32, 150_000], 8, vec![0u8; 150_000 * 34]);
    let p = write(&d, "big.gguf", &b.into_bytes());
    let i = inspect_with(&p, 4096).unwrap();
    assert_eq!(i.parameters, 32 * 150_000);
    assert!(i.file_size > 5_000_000);
    // a header that does not fit in the prefix is reported as exactly that
    let e = inspect_with(&p, 40).unwrap_err();
    assert!(e.contains("big.gguf"), "{e}");
    assert!(e.contains("larger than the 40 bytes"), "{e}");
}

#[test]
fn the_tensor_list_is_capped_but_the_totals_count_everything() {
    let d = tmp();
    let mut b = GgufBuilder::new();
    for i in 0..(MAX_ROWS + 25) {
        b.tensor_f32(&format!("t{i}"), &[1], &[0.0]);
    }
    let info = inspect(&write(&d, "many.gguf", &b.into_bytes())).unwrap();
    assert_eq!(info.n_tensors, MAX_ROWS + 25);
    assert_eq!(info.tensors.len(), MAX_ROWS);
    assert!(info.truncated);
    assert_eq!(info.parameters, (MAX_ROWS + 25) as u64);
    assert_eq!(info.kinds[0].tensors, MAX_ROWS + 25);
}

#[test]
fn a_tensor_type_the_reader_does_not_support_is_refused_not_guessed_at() {
    let d = tmp();
    let mut b = GgufBuilder::new();
    b.tensor_raw("mystery", &[4], 999, vec![0u8; 16]);
    let e = inspect(&write(&d, "odd.gguf", &b.into_bytes())).unwrap_err();
    assert!(
        e.contains("odd.gguf") && e.contains("unsupported ggml tensor type 999"),
        "{e}"
    );
}

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
