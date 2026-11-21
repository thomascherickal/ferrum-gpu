use super::*;
// fills the whole context

#[test]
fn feed_matches_the_oracle_at_every_position_of_a_full_context() {
    let Some(g) = open_for_test() else { return };
    for (name, c, biases, tied) in variants() {
        let w = Llama::random(c, 2, biases, tied);
        let want = w.forward_logits(&IDS).unwrap();
        let mut m = GpuLlama::new(&g, &w, None, None).unwrap();
        for (i, &t) in IDS.iter().enumerate() {
            let e = max_rel_err(&m.feed(t).unwrap(), &want[i * 17..(i + 1) * 17]);
            eprintln!("{name} feed pos {i}: max rel err {e:.2e}");
            assert!(e < FORWARD, "{name} position {i}: {e}");
        }
        assert_eq!(m.pos(), 10);
        assert!(m.feed(1).unwrap_err().to_string().contains("full"));
    }
}

#[test]
fn quantised_feed_matches_the_dequantised_oracle() {
    let Some(g) = open_for_test() else { return };
    for kind in [QKind::Q8_0, QKind::Q4_0] {
        for (name, c, biases, tied) in variants() {
            let w = Llama::random(c, 3, biases, tied);
            let oracle = dequantised(&w, kind).unwrap();
            let want = oracle.forward_logits(&IDS).unwrap();
            let mut m = GpuLlama::new(&g, &w, Some(kind), None).unwrap();
            let mut worst = 0.0f32;
            for (i, &t) in IDS.iter().enumerate() {
                worst = worst.max(max_rel_err(
                    &m.feed(t).unwrap(),
                    &want[i * 17..(i + 1) * 17],
                ));
            }
            eprintln!("{kind:?} {name}: worst feed error {worst:.2e}");
            assert!(worst < FORWARD, "{kind:?} {name}: {worst}");
            // and the quantised model is a real approximation of the f32 one, not the same model
            let exact = w.forward_logits(&IDS).unwrap();
            assert!(
                max_rel_err(&want, &exact) > 1e-4,
                "{kind:?} {name} should differ from f32"
            );
        }
    }
}

#[test]
fn a_model_loaded_from_a_quantised_gguf_runs_identically_to_its_oracle() {
    let Some(g) = open_for_test() else { return };
    for (quant, kind) in [(Quant::Q8_0, QKind::Q8_0), (Quant::Q4_0, QKind::Q4_0)] {
        for (name, c, biases, tied) in variants() {
            let w = Llama::random(c, 4, biases, tied);
            let loaded = from_gguf(&Gguf::parse(to_gguf(&w, quant)).unwrap()).unwrap();
            // requantising a loaded model is the identity on the values the GPU uses
            let again = dequantised(&loaded, kind).unwrap();
            assert_eq!(again.blocks, loaded.blocks, "{name}");
            assert_eq!(again.head, loaded.head, "{name}");
            let want = loaded.forward_logits(&IDS[..6]).unwrap();
            let got = GpuLlama::new(&g, &loaded, Some(kind), None)
                .unwrap()
                .forward_all(&IDS[..6])
                .unwrap();
            let e = max_rel_err(&got, &want);
            eprintln!("gguf {kind:?} {name}: forward_all error {e:.2e}");
            assert!(e < FORWARD, "{kind:?} {name}: {e}");
        }
    }
}

#[test]
fn forward_all_matches_the_oracle_and_feed_for_every_variant() {
    let Some(g) = open_for_test() else { return };
    for (name, c, biases, tied) in variants() {
        let w = Llama::random(c, 5, biases, tied);
        let m = GpuLlama::new(&g, &w, None, None).unwrap();
        for t in [1usize, 2, 7, 10] {
            let want = w.forward_logits(&IDS[..t]).unwrap();
            let e = max_rel_err(&m.forward_all(&IDS[..t]).unwrap(), &want);
            eprintln!("{name} forward_all t={t}: {e:.2e}");
            assert!(e < FORWARD, "{name} t={t}: {e}");
        }
        for bad in [&[][..], &[0; 11], &[17]] {
            assert!(m.forward_all(bad).is_err(), "{name} {bad:?}");
        }
    }
}

#[test]
fn quantised_forward_all_matches_the_dequantised_oracle() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 6, true, false);
    for kind in [QKind::Q8_0, QKind::Q4_0] {
        let want = dequantised(&w, kind)
            .unwrap()
            .forward_logits(&IDS[..8])
            .unwrap();
        let got = GpuLlama::new(&g, &w, Some(kind), None)
            .unwrap()
            .forward_all(&IDS[..8])
            .unwrap();
        let e = max_rel_err(&got, &want);
        eprintln!("{kind:?} forward_all: {e:.2e}");
        assert!(e < FORWARD, "{kind:?}: {e}");
    }
}

#[test]
fn rope_scaling_is_honoured_on_the_device() {
    let Some(g) = open_for_test() else { return };
    let mut w = Llama::random(cfg(), 7, false, false);
    let plain = w.forward_logits(&IDS[..6]).unwrap();
    w.rope = RopeScaling {
        freq_factors: vec![1.0, 2.0, 4.0, 8.0],
        linear_factor: 2.0,
    };
    let want = w.forward_logits(&IDS[..6]).unwrap();
    assert!(max_rel_err(&want, &plain) > 1e-3, "scaling must matter");
    let got = GpuLlama::new(&g, &w, None, None)
        .unwrap()
        .forward_all(&IDS[..6])
        .unwrap();
    assert!(max_rel_err(&got, &want) < FORWARD);
    let mut bad = w.clone();
    bad.rope.linear_factor = 0.0;
    assert!(GpuLlama::new(&g, &bad, None, None)
        .err()
        .expect("refused")
        .to_string()
        .contains("RoPE"));
}

#[test]
fn score_matches_the_oracle() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 8, true, true);
    let ids = [4usize, 8, 15, 16, 0, 1];
    let logits = w.forward_logits(&ids[..5]).unwrap();
    let want = ferrum_ref::nn::score(&logits, &ids[1..], 17);
    let m = GpuLlama::new(&g, &w, None, None).unwrap();
    let got = m.score(&ids).unwrap();
    assert_eq!(got.len(), 5);
    for (i, ((glp, ghit), (wlp, whit))) in got.iter().zip(&want).enumerate() {
        assert!(
            (glp - wlp).abs() < 1e-4 * wlp.abs().max(1.0),
            "row {i}: {glp} vs {wlp}"
        );
        assert_eq!(ghit, whit, "row {i} argmax flag");
    }
    assert!(m.score(&[1]).is_err());
}
