//! The finite-difference check both models' whole-model gradient tests make.
use crate::training::optim::ParamSet;

/// Compare `grad` (the analytic gradient of tensor `pi` of `m`) with central differences of
/// `eval` at up to 6 spread-out elements (all of them when the tensor is small); `name` labels
/// failures. Returns how many elements were checked.
pub(crate) fn check_tensor_fd<M: ParamSet + Clone>(
    m: &M,
    pi: usize,
    grad: &[f32],
    eval: impl Fn(&M) -> f32,
    name: &str,
) -> usize {
    let h = 1e-2;
    let len = m.tensors()[pi].len();
    let picks: Vec<usize> = if len <= 6 {
        (0..len).collect()
    } else {
        (0..6).map(|j| j * (len - 1) / 5).collect()
    };
    for &i in &picks {
        let (mut p, mut q) = (m.clone(), m.clone());
        p.tensors_mut()[pi][i] += h;
        q.tensors_mut()[pi][i] -= h;
        let num = (eval(&p) - eval(&q)) / (2.0 * h);
        let ana = grad[i];
        assert!(
            (ana - num).abs() < 4e-3 + 2e-2 * num.abs(),
            "{name} param {pi}[{i}]: analytic {ana} vs numeric {num}"
        );
    }
    picks.len()
}
