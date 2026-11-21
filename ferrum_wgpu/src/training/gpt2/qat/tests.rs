use super::*;

fn cfg() -> Config {
    Config {
        vocab: 16,
        ctx: 8,
        dim: 32,
        heads: 2,
        layers: 2,
        hidden: 64,
    }
}

#[test]
fn the_snap_plan_matches_the_tensors_a_q8_0_export_quantises() {
    let c = cfg();
    // Count which canonical parameters are quantised and compare with the CPU reference,
    // which snaps exactly the tensors the GGUF writer stores as Q8_0.
    let w = ferrum_ref::gpt2::Gpt2::random(c, 3);
    let snapped = w.fake_quantize();
    for (i, (a, b)) in w.params().iter().zip(snapped.params()).enumerate() {
        let changed = a != &b;
        let planned = snap_kind(&c, i) != Snap::Keep;
        let kind = snap_kind(&c, i);
        assert_eq!(changed, planned, "parameter {i}: kind {kind:?}");
    }
    assert_eq!(snap_kind(&c, 0), Snap::Rows(32));
    assert_eq!(snap_kind(&c, 2 + 12), Snap::Cols(32, 64), "w1");
    assert_eq!(snap_kind(&c, 2 + 14), Snap::Cols(64, 32), "w2");
    assert_eq!(
        snap_kind(&c, 2 + 16 + 2),
        Snap::Cols(32, 32),
        "second block's wq"
    );
    assert_eq!(snap_kind(&c, 2 + 32 + 2), Snap::Cols(32, 16), "head");
}

#[test]
fn the_plan_has_one_buffer_per_parameter() {
    let c = cfg();
    let p = qat_plan(&c);
    assert_eq!(p.rows().len(), 2 + 16 * c.layers + 3);
    let total: u64 = ferrum_ref::gpt2::Gpt2::random(c, 1).num_params() as u64 * 4;
    assert_eq!(p.total(), total);
    // An invalid config yields an empty plan rather than panicking.
    assert!(qat_plan(&Config { heads: 3, ..c }).rows().is_empty());
}
