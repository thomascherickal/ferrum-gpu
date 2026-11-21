use super::*;

#[test]
fn batch_shapes_can_change_between_steps() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 8), None).unwrap();
    let sc = StepConfig {
        adam: Adam::new(0.005),
        clip: 1.0,
    };
    for (b, t) in [(2usize, 5usize), (1, 3), (2, 5), (4, 2)] {
        let (ids, targets) = batch_data(&c, b, t, 0);
        let s = tr.step(&ids, &targets, b, &sc).unwrap();
        assert!(s.loss.is_finite() && s.grad_norm.is_finite(), "{b}x{t}");
    }
}

#[test]
fn bad_input_and_oversized_batches_are_refused_before_allocating() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 9);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    let mut e =
        |ids: &[usize], tg: &[usize], b: usize| tr.step(ids, tg, b, &sc).unwrap_err().to_string();
    assert!(e(&[], &[], 1).contains("split"));
    assert!(e(&[1, 2, 3], &[1, 2, 3], 2).contains("split"));
    assert!(e(&[1; 9], &[1; 9], 1).contains("exceeds the context"));
    assert!(e(&[1, 2], &[1], 1).contains("targets"));
    assert!(e(&[13], &[1], 1).contains("out of range"));
    assert_eq!(tr.steps(), 0, "failed steps do not count");
    // A tight budget: the static part fits, a big batch does not.
    let need = static_plan(&c).total();
    let tight = Some(Budget {
        bytes: need + 1000,
        source: Source::Override,
    });
    let mut small = GpuTrainer::new(&g, &w, tight).unwrap();
    let before = g.buffers_created();
    let (ids, targets) = batch_data(&c, 4, 8, 0);
    let msg = small.step(&ids, &targets, 4, &sc).unwrap_err().to_string();
    assert!(
        msg.contains("does not fit") && msg.contains("budget"),
        "{msg}"
    );
    assert_eq!(
        g.buffers_created(),
        before,
        "refused before any activation buffer exists"
    );
    // And the static part alone can be refused too.
    let tiny = Some(Budget {
        bytes: 100,
        source: Source::Override,
    });
    assert!(GpuTrainer::new(&g, &w, tiny).is_err());
}
