use super::*;

#[test]
fn f32_round_trip_is_exact_for_both_architectures() {
    for (ty, biases, tied) in [
        (RopeType::Norm, false, false),
        (RopeType::Norm, false, true),
        (RopeType::Neox, true, false),
        (RopeType::Neox, true, true),
    ] {
        let m = Llama::random(cfg(ty), 1, biases, tied);
        assert_eq!(
            parse(to_gguf(&m, Quant::F32)).unwrap(),
            m,
            "{ty:?} {biases} {tied}"
        );
    }
}

#[test]
fn the_architecture_decides_the_rope_pairing() {
    let norm = to_gguf(
        &Llama::random(cfg(RopeType::Norm), 1, false, false),
        Quant::F32,
    );
    let neox = to_gguf(
        &Llama::random(cfg(RopeType::Neox), 1, true, false),
        Quant::F32,
    );
    assert_eq!(
        Gguf::parse(norm)
            .unwrap()
            .string("general.architecture")
            .unwrap(),
        "llama"
    );
    assert_eq!(
        Gguf::parse(neox)
            .unwrap()
            .string("general.architecture")
            .unwrap(),
        "qwen2"
    );
}

#[test]
fn round_trip_gives_identical_logits_and_a_tied_file_has_no_output_tensor() {
    let m = Llama::random(cfg(RopeType::Norm), 2, false, true);
    let bytes = to_gguf(&m, Quant::F32);
    let g = Gguf::parse(bytes).unwrap();
    assert!(g.tensor("output.weight").is_none());
    let ids = [3, 1, 4, 1, 5];
    assert_eq!(
        from_gguf(&g).unwrap().forward_logits(&ids).unwrap(),
        m.forward_logits(&ids).unwrap()
    );
    let untied = Llama::random(cfg(RopeType::Norm), 2, false, false);
    assert!(Gguf::parse(to_gguf(&untied, Quant::F32))
        .unwrap()
        .tensor("output.weight")
        .is_some());
}
