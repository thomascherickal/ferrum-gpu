use super::*;

#[test]
fn round_trips_exactly() {
    let cp = sample();
    assert_eq!(from_bytes(to_bytes(&cp)).unwrap(), cp);
    let mut no_id = cp.clone();
    no_id.corpus_id = None;
    assert_eq!(from_bytes(to_bytes(&no_id)).unwrap(), no_id);
}

#[test]
fn names_follow_the_p_i_data_m_v_scheme() {
    let g = Gguf::parse(to_bytes(&sample())).unwrap();
    assert_eq!(g.string("general.architecture").unwrap(), KIND);
    let n = param_lens(&cfg()).unwrap().len();
    assert_eq!(g.tensors.len(), 3 * n);
    for name in ["p0.data", "p0.m", "p0.v", &format!("p{}.v", n - 1)] {
        assert!(g.tensor(name).is_some(), "{name}");
    }
    assert_eq!(g.uint("ferrum.ckpt.step").unwrap(), 42);
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    let good = to_bytes(&sample());
    // Cutting only the final alignment padding leaves a valid file; stop before real data ends.
    let cuts = [0, 1, 10, 100, 1000, good.len() / 2, good.len() - 200];
    for cut in cuts {
        assert!(from_bytes(good[..cut].to_vec()).is_err(), "cut at {cut}");
    }
}

#[test]
fn wrong_kind_and_missing_keys_are_named() {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    assert!(from_bytes(b.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("not a ferrum-gpu checkpoint"));
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(KIND.into()));
    assert!(from_bytes(b.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("missing key"));
}

#[test]
fn a_config_that_lies_about_sizes_is_refused_before_allocating() {
    // Same tensors, but the metadata claims a vastly larger model.
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(KIND.into()));
    for (k, v) in [
        ("vocab", u64::MAX / 4),
        ("ctx", 6),
        ("dim", 8),
        ("heads", 2),
        ("layers", 2),
        ("hidden", 12),
    ] {
        b.meta(&format!("ferrum.ckpt.{k}"), Value::U64(v));
    }
    assert!(from_bytes(b.into_bytes()).is_err());
    // Plausible config, but the file carries no tensors.
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(KIND.into()));
    for (k, v) in [
        ("vocab", 9u64),
        ("ctx", 6),
        ("dim", 8),
        ("heads", 2),
        ("layers", 2),
        ("hidden", 12),
    ] {
        b.meta(&format!("ferrum.ckpt.{k}"), Value::U64(v));
    }
    assert!(from_bytes(b.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("tensor count"));
    // Invalid config (heads do not divide dim).
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str(KIND.into()));
    for (k, v) in [
        ("vocab", 9u64),
        ("ctx", 6),
        ("dim", 8),
        ("heads", 3),
        ("layers", 2),
        ("hidden", 12),
    ] {
        b.meta(&format!("ferrum.ckpt.{k}"), Value::U64(v));
    }
    assert!(from_bytes(b.into_bytes())
        .unwrap_err()
        .to_string()
        .contains("invalid"));
}

#[test]
fn a_tensor_of_the_wrong_length_is_named() {
    let cp = sample();
    let good = Gguf::parse(to_bytes(&cp)).unwrap();
    // Rebuild the file but shorten one moment tensor.
    let mut b = GgufBuilder::new();
    for (k, v) in &good.kv {
        b.meta(k, v.clone());
    }
    for t in &good.tensors {
        let mut data = good.tensor_f32(&t.name).unwrap();
        if t.name == "p4.m" {
            data.pop();
        }
        b.tensor_f32(&t.name, &[data.len() as u64], &data);
    }
    let e = from_bytes(b.into_bytes()).unwrap_err().to_string();
    assert!(e.contains("p4.m") && e.contains("elements"), "{e}");
}

#[test]
fn atomic_write_replaces_the_target_and_leaves_no_temp_file() {
    let d = tmpdir("atomic");
    let p = d.join("run.ckpt");
    atomic_write(&p, b"first").unwrap();
    atomic_write(&p, b"second, longer").unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), b"second, longer");
    let names: Vec<_> = std::fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "no stray temp file: {names:?}");
    assert!(atomic_write(&d.join("no/such/dir/x"), b"x").is_err());
    assert!(atomic_write(Path::new("/"), b"x").is_err());
    let _ = std::fs::remove_dir_all(&d);
}
