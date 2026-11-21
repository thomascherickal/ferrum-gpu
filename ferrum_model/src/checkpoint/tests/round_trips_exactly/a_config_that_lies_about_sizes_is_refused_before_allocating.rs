use super::*;

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
