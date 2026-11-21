use super::*;

#[test]
fn quantised_round_trips_stay_close_and_q8_beats_q4() {
    let m = Llama::random(cfg(RopeType::Neox), 3, true, false);
    let ids = [2, 7, 1, 8];
    let want = m.forward_logits(&ids).unwrap();
    let err = |q| {
        let back = parse(to_gguf(&m, q)).unwrap();
        max_diff(&back.forward_logits(&ids).unwrap(), &want)
    };
    let (e8, e4) = (err(Quant::Q8_0), err(Quant::Q4_0));
    assert!(e8 < 0.3, "q8 err {e8}");
    assert!(e8 < e4, "q8 {e8} should beat q4 {e4}");
    // matrices whose input width is a multiple of 32 are really packed
    let g = Gguf::parse(to_gguf(&m, Quant::Q8_0)).unwrap();
    assert_eq!(
        g.tensor("blk.0.attn_q.weight").unwrap().ggml_type,
        GGML_Q8_0
    );
    assert_eq!(
        g.tensor("blk.0.attn_norm.weight").unwrap().ggml_type,
        GGML_F32
    );
    // a width that is not a multiple of 32 falls back to F32
    let mut c = cfg(RopeType::Norm);
    c.hidden = 48;
    let g = Gguf::parse(to_gguf(&Llama::random(c, 1, false, false), Quant::Q4_0)).unwrap();
    assert_eq!(
        g.tensor("blk.0.ffn_gate.weight").unwrap().ggml_type,
        GGML_Q4_0
    );
    assert_eq!(
        g.tensor("blk.0.ffn_down.weight").unwrap().ggml_type,
        GGML_F32
    );
    c.dim = 8;
    c.head_dim = 2;
    c.rope_dim = 2;
    let g = Gguf::parse(to_gguf(&Llama::random(c, 1, false, false), Quant::Q4_0)).unwrap();
    assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ggml_type, GGML_F32);
}

#[test]
fn rope_scaling_round_trips_through_metadata_and_the_freq_tensor() {
    let mut m = Llama::random(cfg(RopeType::Norm), 4, false, false);
    m.rope = RopeScaling {
        freq_factors: vec![1.0, 2.0, 4.0, 8.0],
        linear_factor: 2.0,
    };
    let back = parse(to_gguf(&m, Quant::F32)).unwrap();
    assert_eq!(back.rope, m.rope);
    assert_eq!(back, m);
}
