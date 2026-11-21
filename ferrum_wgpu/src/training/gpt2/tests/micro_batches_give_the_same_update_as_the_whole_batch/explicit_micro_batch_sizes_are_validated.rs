use super::*;

#[test]
fn explicit_micro_batch_sizes_are_validated() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 54), None).unwrap();
    let (ids, targets) = batch_data(&c, 4, 3, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    for bad in [0usize, 3, 5] {
        let e = tr.step_with_micro(&ids, &targets, 4, bad, &sc).unwrap_err();
        assert!(
            e.to_string().contains("must be positive and divide"),
            "{bad}: {e}"
        );
    }
    assert_eq!(tr.steps(), 0);
    tr.step_with_micro(&ids, &targets, 4, 4, &sc).unwrap();
    assert_eq!(
        tr.last_micro(),
        4,
        "micro equal to the batch is the unsplit path"
    );
}
