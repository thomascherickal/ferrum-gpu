use super::*;

#[test]
fn a_checkpoint_with_a_missing_tensor_is_refused() {
    let good = Gguf::parse(to_bytes(&sample())).unwrap();
    let mut b = GgufBuilder::new();
    for (k, v) in &good.kv {
        b.meta(k, v.clone());
    }
    for t in &good.tensors {
        let data = good.tensor_f32(&t.name).unwrap();
        let name = if t.name == "p4.v" { "q4.v" } else { &t.name };
        b.tensor_f32(name, &[data.len() as u64], &data);
    }
    let e = from_bytes(b.into_bytes()).unwrap_err().to_string();
    assert_eq!(e, "gguf: missing tensor 'p4.v'");
}

#[test]
fn a_tensor_count_that_overflows_is_refused() {
    assert_eq!(tensors_for(4).unwrap(), 12);
    assert_eq!(
        tensors_for(usize::MAX).unwrap_err().to_string(),
        "gguf: tensor count overflows"
    );
}
