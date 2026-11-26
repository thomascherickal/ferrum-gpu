use super::*;

#[test]
fn a_fault_at_any_command_of_a_tied_qat_step_applies_nothing() {
    let Some(g) = open_for_test() else { return };
    let c = qcfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 65), None).unwrap();
    tr.set_tie(true).unwrap();
    tr.set_qat(true).unwrap();
    let (ids, targets) = batch_data(&c, 1, 3, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 1.0,
    };
    let n = g.sweep_faults(|| tr.step(&ids, &targets, 1, &sc).map(|_| ()));
    assert!(n > 21, "{n}");
}
