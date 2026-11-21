use super::*;
use ferrum_gguf::{GGML_F32, GGML_Q8_0};

fn cfg(dim: usize) -> Config {
    Config {
        vocab: 13,
        ctx: 8,
        dim,
        heads: 2,
        layers: 2,
        hidden: dim * 2,
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn transpose_is_correct_and_an_involution() {
    assert_eq!(
        transpose(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    let m: Vec<f32> = (0..12).map(|i| i as f32).collect();
    assert_eq!(transpose(&transpose(&m, 3, 4), 4, 3), m);
}

#[test]
fn f32_round_trip_is_exact() {
    let m = Gpt2::random(cfg(8), 1);
    let back = from_gguf(&Gguf::parse(to_gguf(&m, Quant::F32)).unwrap()).unwrap();
    assert_eq!(back, m);
}

#[test]
fn round_trip_gives_identical_logits() {
    let m = Gpt2::random(cfg(8), 2);
    let back = from_gguf(&Gguf::parse(to_gguf(&m, Quant::F32)).unwrap()).unwrap();
    let ids = [3, 1, 4, 1, 5];
    assert_eq!(
        m.forward_logits(&ids).unwrap(),
        back.forward_logits(&ids).unwrap()
    );
}

#[test]
fn tied_head_is_omitted_and_restored() {
    let mut m = Gpt2::random(cfg(8), 3);
    m.head = transpose(&m.tok, m.cfg.vocab, m.cfg.dim);
    let bytes = to_gguf(&m, Quant::F32);
    let g = Gguf::parse(bytes).unwrap();
    assert!(
        g.tensor("output.weight").is_none(),
        "a tied head is not stored"
    );
    assert_eq!(from_gguf(&g).unwrap(), m);
    // An untied head is stored.
    let untied = Gpt2::random(cfg(8), 3);
    assert!(Gguf::parse(to_gguf(&untied, Quant::F32))
        .unwrap()
        .tensor("output.weight")
        .is_some());
}

#[test]
fn stored_layout_matches_ggml_conventions() {
    let m = Gpt2::random(cfg(8), 4);
    let g = Gguf::parse(to_gguf(&m, Quant::F32)).unwrap();
    assert_eq!(g.tensor("token_embd.weight").unwrap().dims, vec![8, 13]);
    assert_eq!(g.tensor("position_embd.weight").unwrap().dims, vec![8, 8]);
    assert_eq!(g.tensor("blk.0.attn_qkv.weight").unwrap().dims, vec![8, 24]);
    assert_eq!(g.tensor("blk.1.ffn_up.weight").unwrap().dims, vec![8, 16]);
    assert_eq!(g.tensor("blk.1.ffn_down.weight").unwrap().dims, vec![16, 8]);
    assert_eq!(g.tensor("output.weight").unwrap().dims, vec![8, 13]);
    // GGML row 0 of attn_qkv is the first column of Wq (output unit 0 over all inputs).
    let qkv = g.tensor_f32("blk.0.attn_qkv.weight").unwrap();
    let wq = &m.blocks[0].wq;
    assert_eq!(qkv[3], wq[3 * 8]); // [out 0][in 3] = Wq[in 3][out 0]
    assert_eq!(qkv[8 + 2], wq[2 * 8 + 1]); // [out 1][in 2] = Wq[in 2][out 1]
                                           // The K block starts d rows later.
    assert_eq!(qkv[8 * 8 + 5], m.blocks[0].wk[5 * 8]);
}

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

#[test]
fn an_embedding_whose_width_disagrees_with_the_metadata_is_refused() {
    let m = Gpt2::random(cfg(8), 8);
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    for (k, v) in [
        ("context_length", 8u32),
        ("embedding_length", 8),
        ("block_count", 1),
        ("feed_forward_length", 16),
        ("attention.head_count", 2),
    ] {
        b.meta(&format!("gpt2.{k}"), Value::U32(v));
    }
    b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
    // Declared width 8, but the tensor's contiguous dimension is 4.
    b.tensor_f32("token_embd.weight", &[4, 26], &m.tok);
    let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
    assert!(
        e.to_string().contains("must be [embedding_length, vocab]"),
        "{e}"
    );
}

#[test]
fn a_tensor_of_the_wrong_size_is_named() {
    let m = Gpt2::random(cfg(8), 7);
    // Same file but claim heads that do not divide the width.
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    for (k, v) in [
        ("context_length", 8u32),
        ("embedding_length", 8),
        ("block_count", 1),
        ("feed_forward_length", 16),
        ("attention.head_count", 3),
    ] {
        b.meta(&format!("gpt2.{k}"), Value::U32(v));
    }
    b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
    b.tensor_f32("token_embd.weight", &[8, 13], &m.tok);
    let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
    assert!(e.to_string().contains("divisible"), "{e}");
    // Right metadata, truncated position table.
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", Value::Str("gpt2".into()));
    for (k, v) in [
        ("context_length", 8u32),
        ("embedding_length", 8),
        ("block_count", 1),
        ("feed_forward_length", 16),
        ("attention.head_count", 2),
    ] {
        b.meta(&format!("gpt2.{k}"), Value::U32(v));
    }
    b.meta("gpt2.attention.layer_norm_epsilon", Value::F32(LN_EPS));
    b.tensor_f32("token_embd.weight", &[8, 13], &m.tok);
    b.tensor_f32("position_embd.weight", &[8, 2], &m.pos[..16]);
    let e = from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap_err();
    assert!(
        e.to_string().contains("position_embd.weight") && e.to_string().contains("expected 64"),
        "{e}"
    );
}
