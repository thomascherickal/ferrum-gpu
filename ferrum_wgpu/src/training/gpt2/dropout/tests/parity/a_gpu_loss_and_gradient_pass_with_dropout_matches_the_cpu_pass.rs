use super::*;

#[test]
fn a_gpu_loss_and_gradient_pass_with_dropout_matches_the_cpu_pass() {
    use ferrum_ref::dropout::{step_seed, Dropout};
    use ferrum_ref::grad::loss_and_grads_with;
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 5);
    let (ids, targets) = data(&c, 2, 5, 0);
    let mut gpu = GpuTrainer::new(&g, &w, None).unwrap();
    gpu.set_dropout(0.4, 3);
    let loss = gpu.loss_and_grads(&ids, &targets, 2).unwrap();
    let d = Dropout::new(0.4, step_seed(3, 0));
    let (want_loss, want) = loss_and_grads_with(&w, &ids, &targets, 2, 1, d).unwrap();
    assert!(
        (loss - want_loss).abs() < 1e-5 * want_loss.max(1.0),
        "{loss} vs {want_loss}"
    );
    let got = gpu.grads_host().unwrap();
    for (i, (a, b)) in got.params().iter().zip(want.params()).enumerate() {
        let e = max_rel_err(a, b);
        assert!(e < crate::tolerance::BACKWARD, "param {i}: {e}");
    }
}
