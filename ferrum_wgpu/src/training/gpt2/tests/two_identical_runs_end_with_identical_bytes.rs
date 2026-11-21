use super::*;

#[test]
fn two_identical_runs_end_with_identical_bytes() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let run = || {
        let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 6), None).unwrap();
        for s in 0..4usize {
            let (ids, targets) = batch_data(&c, 2, 5, s);
            tr.step(
                &ids,
                &targets,
                2,
                &StepConfig {
                    adam: Adam::new(0.01),
                    clip: 0.5,
                },
            )
            .unwrap();
        }
        (tr.weights_host().unwrap(), tr.moments_host().unwrap())
    };
    assert_eq!(run(), run());
}
