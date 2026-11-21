use super::*;

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
