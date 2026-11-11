use super::*;
use crate::budget::Source;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, FORWARD};
use ferrum_gguf::Gguf;
use ferrum_model::gpt2_gguf::Quant;
use ferrum_model::llama_gguf::{from_gguf, to_gguf};
use ferrum_ref::llama::{Cache, RopeScaling, RopeType};

fn cfg() -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        layers: 2,
        hidden: 64,
        rope_dim: 8,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        eps: 1e-5,
    }
}

/// (name, config, biases, tied): llama-like, qwen-like with biases and a tied head, partial rotary
/// with MQA, and a width that is not a multiple of the quantisation block.
fn variants() -> Vec<(&'static str, Config, bool, bool)> {
    let mut neox = cfg();
    neox.rope_type = RopeType::Neox;
    let mut mqa = cfg();
    (mqa.kv_heads, mqa.rope_dim) = (1, 4);
    let mut odd = cfg();
    (odd.dim, odd.heads, odd.kv_heads, odd.head_dim, odd.rope_dim) = (24, 3, 3, 8, 8);
    vec![
        ("llama", cfg(), false, false),
        ("qwen", neox, true, true),
        ("mqa+partial", mqa, false, false),
        ("dim24", odd, true, false),
    ]
}

const IDS: [usize; 10] = [3, 16, 0, 9, 9, 4, 11, 2, 7, 5]; // fills the whole context

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

#[test]
fn greedy_generation_agrees_with_the_oracle_token_for_token() {
    let Some(g) = open_for_test() else { return };
    for kind in [None, Some(QKind::Q8_0)] {
        let w = Llama::random(cfg(), 9, false, false);
        let oracle = kind.map_or(w.clone(), |k| dequantised(&w, k).unwrap());
        let mut m = GpuLlama::new(&g, &w, kind, None).unwrap();
        let mut cache = Cache::new(2);
        let (mut want, mut got) = (oracle.feed(&mut cache, 2).unwrap(), m.feed(2).unwrap());
        let argmax = |l: &[f32]| (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b });
        for step in 0..7 {
            let (a, b) = (argmax(&want), argmax(&got));
            assert_eq!(a, b, "{kind:?} step {step}");
            want = oracle.feed(&mut cache, a).unwrap();
            got = m.feed(b).unwrap();
        }
    }
}

#[test]
fn reset_restarts_the_context_and_gives_the_same_answers() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 10, false, false);
    let mut m = GpuLlama::new(&g, &w, Some(QKind::Q8_0), None).unwrap();
    let first: Vec<Vec<f32>> = [5usize, 6, 7].iter().map(|&t| m.feed(t).unwrap()).collect();
    m.reset();
    assert_eq!(m.pos(), 0);
    let again: Vec<Vec<f32>> = [5usize, 6, 7].iter().map(|&t| m.feed(t).unwrap()).collect();
    assert_eq!(
        first, again,
        "stale cache entries must not leak into a new context"
    );
    assert_eq!((m.config(), m.quant()), (&cfg(), Some(QKind::Q8_0)));
}

#[test]
fn plan_lists_every_buffer_and_quantisation_shrinks_it() {
    let c = cfg();
    let p = plan(&c, None);
    assert!(p.rows().iter().any(|(n, _)| n == "blk1.kcache"));
    assert!(p.rows().iter().any(|(n, _)| n == "scratch.logits"));
    let kv = f32s(c.ctx * c.kv_dim());
    assert_eq!(
        p.rows().iter().find(|(n, _)| n == "blk0.kcache").unwrap().1,
        kv
    );
    assert!(plan(&Config { layers: 4, ..c }, None).total() > p.total());
    let (q8, q4) = (
        plan(&c, Some(QKind::Q8_0)).total(),
        plan(&c, Some(QKind::Q4_0)).total(),
    );
    assert!(q4 < q8 && q8 < p.total(), "{q4} < {q8} < {}", p.total());
    // an ineligible width keeps its f32 size
    let mut odd = c;
    (odd.dim, odd.heads, odd.kv_heads, odd.head_dim, odd.rope_dim) = (24, 3, 3, 8, 8);
    let wq = |q| {
        plan(&odd, q)
            .rows()
            .iter()
            .find(|(n, _)| n == "blk0.wq")
            .unwrap()
            .1
    };
    assert_eq!(wq(Some(QKind::Q8_0)), wq(None));
}

#[test]
fn the_plan_matches_what_is_really_uploaded() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 11, false, false);
    for q in [None, Some(QKind::Q8_0), Some(QKind::Q4_0)] {
        let m = GpuLlama::new(&g, &w, q, None).unwrap();
        let planned = |name: &str| {
            plan(&cfg(), q)
                .rows()
                .iter()
                .find(|(n, _)| n == name)
                .unwrap()
                .1
        };
        assert_eq!(
            m.blocks[0].wq.device_bytes(),
            planned("blk0.wq"),
            "{q:?} wq"
        );
        assert_eq!(
            m.blocks[1].down.device_bytes(),
            planned("blk1.down"),
            "{q:?} down"
        );
        assert_eq!(m.head.device_bytes(), planned("head"), "{q:?} head");
    }
}

#[test]
fn an_oversized_model_is_refused_before_allocating() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 1, false, false);
    let before = g.buffers_created();
    let tiny = Budget {
        bytes: 1000,
        source: Source::Override,
    };
    let e = GpuLlama::new(&g, &w, None, Some(tiny))
        .err()
        .expect("must refuse");
    assert!(e.to_string().contains("budget is 1000"), "{e}");
    assert_eq!(g.buffers_created(), before);
    // quantisation can bring a model under a budget it would otherwise miss
    let f32_total = plan(&cfg(), None).total();
    let q4_total = plan(&cfg(), Some(QKind::Q4_0)).total();
    let between = Budget {
        bytes: (f32_total + q4_total) / 2,
        source: Source::Override,
    };
    assert!(GpuLlama::new(&g, &w, None, Some(between)).is_err());
    assert!(GpuLlama::new(&g, &w, Some(QKind::Q4_0), Some(between)).is_ok());
}

#[test]
fn a_fault_at_any_command_fails_cleanly_and_the_context_does_not_advance() {
    let Some(g) = open_for_test() else { return };
    let w = Llama::random(cfg(), 13, false, false);
    let mut m = GpuLlama::new(&g, &w, None, None).unwrap();
    let commands = g.sweep_faults(|| {
        let before = m.pos();
        let r = m.feed(3).map(|_| ());
        if r.is_err() {
            assert_eq!(
                m.pos(),
                before,
                "a failed token must not advance the position"
            );
        }
        r
    });
    // 2 (index copy + gather) + 18 per block x 2 blocks + 2 (final norm + head).
    assert_eq!(commands, 40);
    assert_eq!(m.pos(), 1, "only the successful attempt advanced");
    let n = g.sweep_faults(|| m.forward_all(&[1, 2, 3]).map(|_| ()));
    // 1 gather + 17 per block x 2 (no cache copies, a 3-dispatch attention) + final norm + head.
    assert_eq!(n, 37);
    // score = the batched pass over all but the last token (1 + 17 x 2 + 2) + the score kernel.
    assert_eq!(
        g.sweep_faults(|| m.score(&[1, 2, 3, 4]).map(|_| ())),
        37 + 1
    );
    // the quantised path with biases records extra bias dispatches
    let wb = Llama::random(cfg(), 14, true, false);
    let mut mq = GpuLlama::new(&g, &wb, Some(QKind::Q8_0), None).unwrap();
    let nq = g.sweep_faults(|| mq.feed(3).map(|_| ()));
    assert_eq!(
        nq,
        40 + 3 * 2,
        "one extra add_bias per biased q/k/v per block"
    );
}
