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
