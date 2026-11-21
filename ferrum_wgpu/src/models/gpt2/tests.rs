use super::*;
use crate::device::open_for_test;
use crate::tolerance::{max_rel_err, FORWARD};
use ferrum_ref::gpt2::Gpt2;

fn cfg() -> Config {
    Config {
        vocab: 17,
        ctx: 10,
        dim: 12,
        heads: 3,
        layers: 2,
        hidden: 20,
    }
}

#[test]
fn a_fault_at_any_command_fails_decoding_cleanly_and_the_context_does_not_advance() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 13);
    let mut m = GpuGpt2::new(&g, &w, None).unwrap();
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
    // 2 (index copy + embed) + 15 per block x 2 blocks + 2 (final norm + head).
    assert_eq!(commands, 34);
    assert_eq!(m.pos(), 1, "only the successful attempt advanced");
    // Batched prefill and scoring share the same recording path.
    let n = g.sweep_faults(|| m.forward_all(&[1, 2, 3]).map(|_| ()));
    assert!(n > 40, "{n}");
    g.sweep_faults(|| m.score(&[1, 2, 3, 4]).map(|_| ()));
}

#[test]
fn the_decoder_reports_its_configuration() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 12);
    let m = GpuGpt2::new(&g, &w, None).unwrap();
    assert_eq!(m.config(), &cfg());
    assert_eq!(m.pos(), 0);
}

#[test]
fn plan_lists_every_buffer_and_scales_with_the_model() {
    let c = cfg();
    let p = plan(&c);
    assert!(p.rows().iter().any(|(n, _)| n == "blk1.kcache"));
    assert!(p.rows().iter().any(|(n, _)| n == "scratch.logits"));
    let kv = f32s(c.ctx * c.dim);
    assert_eq!(
        p.rows().iter().find(|(n, _)| n == "blk0.kcache").unwrap().1,
        kv
    );
    let bigger = plan(&Config { layers: 4, ..c });
    assert!(bigger.total() > p.total());
    assert_eq!(bigger.rows().len() - p.rows().len(), 2 * 18);
}

#[test]
fn an_oversized_model_is_refused_before_allocating() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 1);
    let before = g.buffers_created();
    let tiny = Budget {
        bytes: 1000,
        source: crate::budget::Source::Override,
    };
    let e = GpuGpt2::new(&g, &w, Some(tiny)).err().expect("must refuse");
    assert!(e.to_string().contains("budget is 1000"), "{e}");
    assert_eq!(g.buffers_created(), before);
}

#[test]
fn feed_matches_the_oracle_logits_row_by_row() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 2);
    let ids = [3usize, 16, 0, 9, 9, 4, 11, 2, 7, 5]; // fills the whole context
    let want = w.forward_logits(&ids).unwrap();
    let mut m = GpuGpt2::new(&g, &w, None).unwrap();
    for (i, &t) in ids.iter().enumerate() {
        let got = m.feed(t).unwrap();
        let e = max_rel_err(&got, &want[i * 17..(i + 1) * 17]);
        eprintln!("feed pos {i}: max rel err {e:.2e}");
        assert!(e < FORWARD, "position {i}: {e}");
    }
    assert_eq!(m.pos(), 10);
    let full = m.feed(1).unwrap_err();
    assert!(full.to_string().contains("full"));
}

#[test]
fn reset_restarts_the_context_and_gives_the_same_answers() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 3);
    let mut m = GpuGpt2::new(&g, &w, None).unwrap();
    let first: Vec<Vec<f32>> = [5usize, 6, 7].iter().map(|&t| m.feed(t).unwrap()).collect();
    m.reset();
    assert_eq!(m.pos(), 0);
    let again: Vec<Vec<f32>> = [5usize, 6, 7].iter().map(|&t| m.feed(t).unwrap()).collect();
    assert_eq!(
        first, again,
        "stale cache entries must not leak into a new context"
    );
}

#[test]
fn forward_all_matches_the_oracle_and_feed() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 4);
    let ids = [1usize, 2, 3, 4, 5, 6, 7];
    let want = w.forward_logits(&ids).unwrap();
    let m = GpuGpt2::new(&g, &w, None).unwrap();
    let e = max_rel_err(&m.forward_all(&ids).unwrap(), &want);
    eprintln!("forward_all: max rel err {e:.2e}");
    assert!(e < FORWARD);
    assert!(m
        .forward_all(&[])
        .unwrap_err()
        .to_string()
        .contains("sequence length"));
    assert!(m.forward_all(&[0; 11]).is_err());
    assert!(m.forward_all(&[17]).is_err());
}

#[test]
fn score_matches_the_oracle() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 5);
    let ids = [4usize, 8, 15, 16, 0, 1];
    let logits = w.forward_logits(&ids[..5]).unwrap();
    let want = ferrum_ref::nn::score(&logits, &ids[1..], 17);
    let got = GpuGpt2::new(&g, &w, None).unwrap().score(&ids).unwrap();
    assert_eq!(got.len(), 5);
    for (i, ((glp, ghit), (wlp, whit))) in got.iter().zip(&want).enumerate() {
        assert!(
            (glp - wlp).abs() < 1e-4 * wlp.abs().max(1.0),
            "row {i}: {glp} vs {wlp}"
        );
        assert_eq!(ghit, whit, "row {i} argmax flag");
    }
    assert!(GpuGpt2::new(&g, &w, None).unwrap().score(&[1]).is_err());
}

#[test]
fn greedy_generation_matches_the_oracle_token_for_token() {
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 6);
    let mut m = GpuGpt2::new(&g, &w, None).unwrap();
    let mut ids = vec![2usize];
    let mut logits = m.feed(2).unwrap();
    for _ in 0..6 {
        let next = (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
        ids.push(next);
        // The oracle sees the same history and must agree on the last row's argmax.
        let o = w.forward_logits(&ids[..ids.len() - 1]).unwrap();
        let last = &o[(ids.len() - 2) * 17..];
        let want = (0..17).fold(0, |b, i| if last[i] > last[b] { i } else { b });
        assert_eq!(next, want, "history {ids:?}");
        logits = m.feed(next).unwrap();
    }
}

#[test]
fn a_gguf_exported_model_runs_identically_after_a_round_trip() {
    use ferrum_gguf::Gguf;
    use ferrum_model::gpt2_gguf::{from_gguf, to_gguf, Quant};
    let Some(g) = open_for_test() else { return };
    let w = Gpt2::random(cfg(), 7);
    let loaded = from_gguf(&Gguf::parse(to_gguf(&w, Quant::F32)).unwrap()).unwrap();
    let ids = [1usize, 5, 9];
    let (mut a, mut b) = (
        GpuGpt2::new(&g, &w, None).unwrap(),
        GpuGpt2::new(&g, &loaded, None).unwrap(),
    );
    for &t in &ids {
        assert_eq!(a.feed(t).unwrap(), b.feed(t).unwrap());
    }
}
