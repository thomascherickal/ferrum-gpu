use super::*;

#[test]
fn attention_refuses_bad_shapes() {
    let Some(g) = open_for_test() else { return };
    let z = g.zeros(48).unwrap();
    assert!(attention_gqa(&g, (&z, &z, &z), (&z, &z), (1, 2, 3, 2, 4))
        .unwrap_err()
        .to_string()
        .contains("multiple"));
    assert!(attention_gqa(&g, (&z, &z, &z), (&z, &z), (1, 2, 2, 0, 4)).is_err());
    assert!(attention_gqa(&g, (&z, &z, &z), (&z, &z), (1, 2, 2, 2, 0)).is_err());
    assert!(
        attention_gqa(&g, (&z, &z, &z), (&z, &z), (1, 3, 2, 2, 4)).is_err(),
        "buffer sizes are checked"
    );
    let kc = g.zeros(4 * 8).unwrap();
    let (q, pr, o) = (
        g.zeros(16).unwrap(),
        g.zeros(16).unwrap(),
        g.zeros(16).unwrap(),
    );
    let go = |heads, kv, len| {
        attention_decode_gqa(&g, (&q, &kc, &kc), (&pr, &o), (heads, kv, 4, len, 4))
    };
    assert!(go(4, 2, 0).unwrap_err().to_string().contains("len 0"));
    assert!(go(4, 2, 5).is_err());
    assert!(go(4, 3, 2).is_err());
    assert!(go(4, 2, 4).is_ok());
}

#[test]
fn multi_command_ops_fail_cleanly_at_every_command() {
    let Some(g) = open_for_test() else { return };
    let (batch, t, heads, kv, hd) = (1, 3, 4, 2, 4);
    let (q, k) = (
        g.zeros(t * heads * hd).unwrap(),
        g.zeros(t * kv * hd).unwrap(),
    );
    let (p, o) = (
        g.zeros(heads * t * t).unwrap(),
        g.zeros(t * heads * hd).unwrap(),
    );
    assert_eq!(
        g.sweep_faults(|| attention_gqa(&g, (&q, &k, &k), (&p, &o), (batch, t, heads, kv, hd))),
        3
    );
    let (qd, kc) = (g.zeros(heads * hd).unwrap(), g.zeros(4 * kv * hd).unwrap());
    let (pr, od) = (g.zeros(heads * 4).unwrap(), g.zeros(heads * hd).unwrap());
    assert_eq!(
        g.sweep_faults(|| attention_decode_gqa(
            &g,
            (&qd, &kc, &kc),
            (&pr, &od),
            (heads, kv, hd, 2, 4)
        )),
        2
    );
    let (cs, sn) = (g.zeros(8).unwrap(), g.zeros(8).unwrap());
    let x = g.zeros(2 * heads * hd).unwrap();
    let xo = g.zeros(2 * heads * hd).unwrap();
    assert_eq!(
        g.sweep_faults(|| rope(&g, &x, (&cs, &sn), (2, heads, hd, 4), RopeType::Neox, 0)),
        1
    );
    assert_eq!(g.sweep_faults(|| swiglu(&g, &x, &x, &xo)), 1);
    let (gain, y) = (g.zeros(4).unwrap(), g.zeros(8).unwrap());
    let x8 = g.zeros(8).unwrap();
    assert_eq!(
        g.sweep_faults(|| rmsnorm(&g, &x8, &gain, &y, (2, 4), 1e-5)),
        1
    );
    let (ids, out) = (g.upload_indices(&[0, 1], 2).unwrap(), g.zeros(8).unwrap());
    assert_eq!(
        g.sweep_faults(|| gather_rows(&g, &x8, &ids, &out, (2, 4))),
        1
    );
}
