use super::*;

#[test]
pub(super) fn whole_model_gradients_match_finite_differences_for_every_parameter_group() {
    for (name, m) in variants() {
        let (batch, t) = (2, 4);
        let (ids, targets) = data(&m.cfg, batch, t);
        let (loss, g) = loss_and_grads(&m, &ids, &targets, batch, 1).unwrap();
        assert!(loss.is_finite() && loss > 0.0);
        let eval = |m: &Llama| loss_and_grads(m, &ids, &targets, batch, 1).unwrap().0;
        let mut checked = 0;
        for pi in 0..m.params().len() {
            if m.params()[pi].is_empty() {
                assert!(
                    g.params()[pi].is_empty(),
                    "{name}: an absent bias has no gradient"
                );
                continue;
            }
            checked += crate::grad::check_tensor_fd(&m, pi, g.params()[pi], eval, name);
        }
        assert!(checked > 100, "{name}: only {checked} elements checked");
    }
}
