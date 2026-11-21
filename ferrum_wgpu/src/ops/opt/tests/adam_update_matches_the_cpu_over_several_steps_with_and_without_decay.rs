use super::*;

#[test]
fn adam_update_matches_the_cpu_over_several_steps_with_and_without_decay() {
    let Some(g) = open_for_test() else { return };
    let cfg = Config {
        vocab: 9,
        ctx: 6,
        dim: 8,
        heads: 2,
        layers: 1,
        hidden: 12,
    };
    let w0 = Gpt2::random(cfg, 7);
    let adam = Adam {
        weight_decay: 0.05,
        ..Adam::new(0.01)
    };
    let mut cpu = w0.clone();
    let mut st = AdamState::new(&w0);
    let up = |m: &Gpt2| {
        m.params()
            .iter()
            .map(|p| g.upload(p).unwrap())
            .collect::<Vec<_>>()
    };
    let (gp, gm, gv) = (up(&w0), up(&w0.zeros_like()), up(&w0.zeros_like()));
    let scale = g.upload(&[0.5, 0.0]).unwrap();
    for step in 1..=4u64 {
        let grads = Gpt2::random(cfg, 100 + step);
        adam_step(&mut cpu, &grads, &mut st, &adam, 0.5);
        let gg = up(&grads);
        for i in 0..gp.len() {
            adam_update(
                &g,
                (&gp[i], &gg[i], &gm[i], &gv[i]),
                &scale,
                &adam,
                w0.decays(i),
                step,
            )
            .unwrap();
        }
    }
    for (i, want) in cpu.params().iter().enumerate() {
        let got = g.read(&gp[i]).unwrap();
        let e = max_rel_err(&got, want);
        assert!(e < 1e-5, "param {i}: max rel err {e}");
    }
    for (i, want) in st.v.params().iter().enumerate() {
        assert!(max_rel_err(&g.read(&gv[i]).unwrap(), want) < 1e-5, "v {i}");
    }
    let _ = global_norm(&cpu);
}

#[test]
fn errors_are_reported_not_dispatched() {
    let Some(g) = open_for_test() else { return };
    let (a, b) = (g.zeros(4).unwrap(), g.zeros(3).unwrap());
    let s = g.zeros(2).unwrap();
    assert!(adam_update(&g, (&a, &b, &a, &a), &s, &Adam::new(0.1), true, 1).is_err());
    assert!(
        adam_update(&g, (&a, &a, &a, &a), &s, &Adam::new(0.1), true, 0)
            .unwrap_err()
            .to_string()
            .contains("at least 1")
    );
    let part = g.zeros(1).unwrap();
    assert!(
        sumsq_parts(&g, &a, &part, 1).is_err(),
        "offset beyond the buffer"
    );
    assert!(norm_scale(&g, &part, &g.zeros(1).unwrap(), 1, 1.0).is_err());
    assert!(norm_scale(&g, &part, &s, 5, 1.0).is_err());
}
