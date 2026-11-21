use super::*;

#[test]
fn two_full_gpu_runs_are_byte_identical() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let toks = pattern(300, c.vocab);
    let run = || {
        let mut t = GpuTrainer::new(&g, &Gpt2::init(c, 23), None).unwrap();
        let r = train(
            &mut t,
            &toks,
            &TrainConfig {
                max_steps: 25,
                ..loop_cfg()
            },
            None,
            None,
            &mut NoHooks,
        )
        .unwrap();
        (r.losses, t.snapshot().unwrap())
    };
    assert_eq!(run(), run());
}
