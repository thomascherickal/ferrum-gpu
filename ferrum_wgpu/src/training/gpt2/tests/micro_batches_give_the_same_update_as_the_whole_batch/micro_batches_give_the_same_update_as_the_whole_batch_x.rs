use super::*;

#[test]
fn micro_batches_give_the_same_update_as_the_whole_batch() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w0 = Gpt2::init(c, 51);
    let sc = StepConfig {
        adam: Adam {
            weight_decay: 0.01,
            ..Adam::new(0.01)
        },
        clip: 0.3,
    };
    let steps: Vec<_> = (0..3).map(|s| batch_data(&c, 6, 5, s)).collect();
    let mut whole = GpuTrainer::new(&g, &w0, None).unwrap();
    let whole_stats: Vec<StepStats> = steps
        .iter()
        .map(|(i, t)| whole.step(i, t, 6, &sc).unwrap())
        .collect();
    assert_eq!(whole.last_micro(), 6, "a batch that fits is not split");
    for micro in [1usize, 2, 3] {
        let mut split = GpuTrainer::new(&g, &w0, None).unwrap();
        for (s, (ids, tg)) in steps.iter().enumerate() {
            let st = split.step_with_micro(ids, tg, 6, micro, &sc).unwrap();
            let want = &whole_stats[s];
            assert!(
                (st.loss - want.loss).abs() < 1e-5 * want.loss.max(1.0),
                "micro {micro} step {s}: loss {} vs {}",
                st.loss,
                want.loss
            );
            assert!(
                (st.grad_norm - want.grad_norm).abs() < 1e-4 * want.grad_norm.max(1.0),
                "micro {micro} step {s}: norm"
            );
        }
        assert_eq!(split.last_micro(), micro);
        let e = max_err_models(
            &split.weights_host().unwrap(),
            &whole.weights_host().unwrap(),
        );
        eprintln!("micro {micro}: weights after 3 steps differ by {e:.2e}");
        assert!(e < crate::tolerance::BACKWARD, "micro {micro}: {e}");
    }
}
