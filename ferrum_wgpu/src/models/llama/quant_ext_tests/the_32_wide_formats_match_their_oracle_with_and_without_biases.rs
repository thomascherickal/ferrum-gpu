use super::*;

#[test]
fn the_32_wide_formats_match_their_oracle_with_and_without_biases() {
    let Some(g) = open_for_test() else { return };
    for kind in [QKind::Fp4, QKind::Fp8E4M3, QKind::Fp8E5M2] {
        for (biases, tied) in [(false, false), (true, false), (true, true)] {
            let w = Llama::random(cfg(32, 64, 8), 3, biases, tied);
            let e = feed_error(&g, &w, kind);
            assert!(e < FORWARD, "{kind:?} biases {biases} tied {tied}: {e}");
            // a real approximation, not the f32 model
            let exact = w.forward_logits(&IDS).unwrap();
            let want = dequantised(&w, kind).unwrap().forward_logits(&IDS).unwrap();
            assert!(
                max_rel_err(&want, &exact) > 1e-4,
                "{kind:?} should differ from f32"
            );
        }
    }
}

#[test]
fn q2_k_matches_its_oracle_on_whole_blocks_and_on_the_q4_0_fallback() {
    let Some(g) = open_for_test() else { return };
    // 256 wide, hidden 512: every matrix is Q2_K. hidden 704 (22 x 32): the down projection falls
    // back to Q4_0 and the others stay Q2_K.
    for (name, c) in [
        ("whole", cfg(256, 512, 64)),
        ("fallback", cfg(256, 704, 64)),
    ] {
        let w = Llama::random(c, 5, true, false);
        let e = feed_error(&g, &w, QKind::Q2K);
        eprintln!("Q2_K {name} on {}: worst feed error {e:.2e}", g.info().name);
        assert!(e < Q2K_FORWARD, "{name}: {e}");
    }
    // the 32-wide model has no 256-wide matrix: all of it falls back to Q4_0, i.e. equals Q4_0
    let w = Llama::random(cfg(32, 64, 8), 6, false, false);
    assert_eq!(
        dequantised(&w, QKind::Q2K).unwrap().blocks,
        dequantised(&w, QKind::Q4_0).unwrap().blocks
    );
}

#[test]
fn the_fallback_follows_the_width() {
    assert_eq!(effective(QKind::Q2K, 256), QKind::Q2K);
    assert_eq!(effective(QKind::Q2K, 512), QKind::Q2K);
    assert_eq!(effective(QKind::Q2K, 704), QKind::Q4_0);
    assert_eq!(
        effective(QKind::Q2K, 100),
        QKind::Q2K,
        "not even 32: stays, and is then f32"
    );
    assert_eq!(effective(QKind::Fp4, 704), QKind::Fp4);
    assert_eq!(effective(QKind::Q8_0, 100), QKind::Q8_0);
    assert!(packed_words(effective(QKind::Q2K, 704), 704, 3).is_some());
    assert!(packed_words(effective(QKind::Q2K, 100), 100, 3).is_none());
}

#[test]
fn a_model_loaded_from_an_f16_or_mxfp4_file_runs_identically_to_its_oracle() {
    let Some(g) = open_for_test() else { return };
    for (quant, kind) in [(Quant::F16, QKind::F16), (Quant::Fp4, QKind::Fp4)] {
        let w = Llama::random(cfg(32, 64, 8), 4, true, false);
        let loaded = from_gguf(&Gguf::parse(to_gguf(&w, quant)).unwrap()).unwrap();
        // requantising the loaded model is the identity on the values the GPU uses
        let again = dequantised(&loaded, kind).unwrap();
        assert_eq!(again.blocks, loaded.blocks, "{kind:?}");
        assert_eq!(again.head, loaded.head, "{kind:?}");
        let want = loaded.forward_logits(&IDS[..6]).unwrap();
        let got = GpuLlama::new(&g, &loaded, Some(kind), None)
            .unwrap()
            .forward_all(&IDS[..6])
            .unwrap();
        let e = max_rel_err(&got, &want);
        assert!(e < FORWARD, "{kind:?}: {e}");
    }
}
