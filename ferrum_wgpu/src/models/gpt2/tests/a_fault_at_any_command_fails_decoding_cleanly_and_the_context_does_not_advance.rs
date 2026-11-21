use super::*;

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
