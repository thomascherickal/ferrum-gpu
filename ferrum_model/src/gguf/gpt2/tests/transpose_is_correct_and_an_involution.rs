use super::*;

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
