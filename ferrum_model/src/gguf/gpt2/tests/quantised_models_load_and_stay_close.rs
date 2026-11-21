use super::*;

#[test]
fn quantised_models_load_and_stay_close() {
    let m = Gpt2::random(cfg(32), 5);
    let ids = [1, 7, 3, 9];
    let want = m.forward_logits(&ids).unwrap();
    let q8 = from_gguf(&Gguf::parse(to_gguf(&m, Quant::Q8_0)).unwrap()).unwrap();
    let q4 = from_gguf(&Gguf::parse(to_gguf(&m, Quant::Q4_0)).unwrap()).unwrap();
    let (e8, e4) = (
        max_diff(&q8.forward_logits(&ids).unwrap(), &want),
        max_diff(&q4.forward_logits(&ids).unwrap(), &want),
    );
    assert!(e8 < 0.3, "q8 logits drift {e8}");
    assert!(e8 < e4, "q4 ({e4}) must be coarser than q8 ({e8})");
    // Ineligible widths (dim 8) stay F32 even when Q8_0 is requested.
    let small = Gpt2::random(cfg(8), 5);
    let g = Gguf::parse(to_gguf(&small, Quant::Q8_0)).unwrap();
    assert_eq!(g.tensor("blk.0.ffn_up.weight").unwrap().ggml_type, GGML_F32);
    // A 32-wide model stores its matrices quantised.
    let g32 = Gguf::parse(to_gguf(&m, Quant::Q8_0)).unwrap();
    assert_eq!(
        g32.tensor("blk.0.ffn_up.weight").unwrap().ggml_type,
        GGML_Q8_0
    );
    assert_eq!(
        g32.tensor("blk.0.attn_norm.weight").unwrap().ggml_type,
        GGML_F32
    );
}

#[test]
fn a_qat_snapped_model_survives_a_q8_0_export_exactly() {
    // The point of quantisation-aware training: after the snap, exporting as Q8_0 and
    // reloading loses nothing, because the snap and the export use the same quantiser.
    let snapped = Gpt2::random(cfg(32), 11).fake_quantize();
    let back = from_gguf(&Gguf::parse(to_gguf(&snapped, Quant::Q8_0)).unwrap()).unwrap();
    assert_eq!(back, snapped, "every tensor, bit for bit");
    // An unsnapped model of the same shape does lose precision on export.
    let raw = Gpt2::random(cfg(32), 11);
    let lossy = from_gguf(&Gguf::parse(to_gguf(&raw, Quant::Q8_0)).unwrap()).unwrap();
    assert_ne!(lossy, raw);
    assert_eq!(
        lossy,
        raw.fake_quantize(),
        "the export applies exactly the QAT snap"
    );
    // And the snapped model computes the same logits before and after the round trip.
    let ids = [3, 1, 4, 1, 5];
    assert_eq!(
        snapped.forward_logits(&ids).unwrap(),
        back.forward_logits(&ids).unwrap()
    );
}

#[test]
fn rejects_foreign_or_inconsistent_files() {
    let m = Gpt2::random(cfg(8), 6);
    let ok = to_gguf(&m, Quant::F32);
    // wrong architecture
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("llama".into()));
    assert!(from_gguf(&Gguf::parse(b.into_bytes()).unwrap())
        .unwrap_err()
        .to_string()
        .contains("only 'gpt2'"));
    // missing metadata and tensors
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    assert!(from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).is_err());
    // a different epsilon
    let mut bytes = ok.clone();
    let eps = LN_EPS.to_bits().to_le_bytes();
    let at = bytes
        .windows(4)
        .position(|w| w == eps)
        .expect("epsilon bytes present");
    bytes[at..at + 4].copy_from_slice(&1e-6f32.to_bits().to_le_bytes());
    assert!(from_gguf(&Gguf::parse(bytes).unwrap())
        .unwrap_err()
        .to_string()
        .contains("layer_norm_epsilon"));
}
