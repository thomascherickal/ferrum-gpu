use super::*;

#[test]
pub(super) fn whole_model_gradients_match_finite_differences_for_every_parameter_group() {
    for (name, m) in variants() {
        let (batch, t) = (2, 4);
        let (ids, targets) = data(&m.cfg, batch, t);
        let (loss, g) = loss_and_grads(&m, &ids, &targets, batch, 1).unwrap();
        assert!(loss.is_finite() && loss > 0.0);
        let eval = |m: &Llama| loss_and_grads(m, &ids, &targets, batch, 1).unwrap().0;
        let h = 1e-2;
        let mut checked = 0;
        for pi in 0..m.params().len() {
            let len = m.params()[pi].len();
            if len == 0 {
                assert!(
                    g.params()[pi].is_empty(),
                    "{name}: an absent bias has no gradient"
                );
                continue;
            }
            let picks: Vec<usize> = if len <= 6 {
                (0..len).collect()
            } else {
                (0..6).map(|j| j * (len - 1) / 5).collect()
            };
            for &i in &picks {
                let (mut p, mut q) = (m.clone(), m.clone());
                p.params_mut()[pi][i] += h;
                q.params_mut()[pi][i] -= h;
                let num = (eval(&p) - eval(&q)) / (2.0 * h);
                let ana = g.params()[pi][i];
                assert!(
                    (ana - num).abs() < 4e-3 + 2e-2 * num.abs(),
                    "{name} param {pi}[{i}]: analytic {ana} vs numeric {num}"
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "{name}: only {checked} elements checked");
    }
}
