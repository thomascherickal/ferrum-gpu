use super::*;

#[test]
fn no_dropout_is_bit_identical_to_the_plain_pass() {
    let m = Gpt2::init(cfg(), 3);
    let (ids, targets) = batch();
    let (l0, g0) = loss_and_grads(&m, &ids, &targets, 2, 1).unwrap();
    let (l1, g1) = run(&m, None);
    assert_eq!((l0, g0), (l1, g1));
}

#[test]
fn the_same_seed_reproduces_the_pass_and_another_seed_changes_it() {
    let m = Gpt2::init(cfg(), 3);
    let d = |seed| Dropout::new(0.3, seed);
    assert_eq!(run(&m, d(5)), run(&m, d(5)));
    assert_ne!(
        run(&m, d(5)).0,
        run(&m, d(6)).0,
        "a different mask, a different loss"
    );
    assert_ne!(run(&m, d(5)).0, run(&m, None).0);
}

#[test]
fn the_gradient_is_that_of_the_masked_network() {
    let m = Gpt2::init(cfg(), 4);
    let d = Dropout::new(0.4, 11);
    let (_, g) = run(&m, d);
    // a first-block and a last-block FFN weight, and an embedding row that is read
    let probes = [
        Probe {
            name: "blocks[0].w1[5]",
            bump: |m, e| m.blocks[0].w1[5] += e,
            analytic: g.blocks[0].w1[5],
        },
        Probe {
            name: "blocks[0].w2[9]",
            bump: |m, e| m.blocks[0].w2[9] += e,
            analytic: g.blocks[0].w2[9],
        },
        Probe {
            name: "blocks[1].w1[40]",
            bump: |m, e| m.blocks[1].w1[40] += e,
            analytic: g.blocks[1].w1[40],
        },
        Probe {
            name: "tok[3*8+2]",
            bump: |m, e| m.tok[3 * 8 + 2] += e,
            analytic: g.tok[3 * 8 + 2],
        },
    ];
    for p in &probes {
        let n = numeric(&m, d, p, 5e-3);
        assert!(
            (n - p.analytic).abs() <= 3e-3 + 0.05 * p.analytic.abs(),
            "{}: numeric {n} vs analytic {}",
            p.name,
            p.analytic
        );
    }
}

#[test]
fn a_dropped_unit_receives_no_gradient() {
    // with nearly everything dropped, most FFN input-weight gradients vanish (dropped units pass
    // nothing back through GELU), unlike the undropped network
    let m = Gpt2::init(cfg(), 4);
    let (_, plain) = run(&m, None);
    let (_, heavy) = run(&m, Dropout::new(0.95, 2));
    let zeros = |g: &Gpt2| g.blocks[0].b1.iter().filter(|&&x| x == 0.0).count();
    let (z_heavy, z_plain) = (zeros(&heavy), zeros(&plain));
    assert!(z_heavy > z_plain, "{z_heavy} vs {z_plain}");
}
