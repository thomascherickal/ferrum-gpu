use super::*;

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
