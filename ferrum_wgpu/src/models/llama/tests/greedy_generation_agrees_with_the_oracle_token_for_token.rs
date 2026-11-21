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
