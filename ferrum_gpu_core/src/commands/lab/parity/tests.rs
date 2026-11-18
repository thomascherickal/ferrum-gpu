use super::*;

#[test]
fn every_check_passes_on_an_available_adapter() {
    let Some(gpu) = ferrum_wgpu::device::open_for_test() else {
        return;
    };
    let checks = run(&gpu).unwrap();
    // f32 prefill, decode, six precisions on the small model, Q2_K on the wide one, blocks
    assert_eq!(checks.len(), 10, "{checks:?}");
    for c in &checks {
        assert!(
            c.ok(),
            "{} error {} (allowed {})",
            c.name,
            c.error,
            c.tolerance
        );
        assert_eq!(c.tolerance, FORWARD);
    }
    assert!(checks.iter().any(|c| c.name.contains("blocks")));
    assert!(checks.iter().any(|c| c.name.contains("Q4_0")));
    assert!(checks.iter().any(|c| c.name.contains("F16")));
}

#[test]
fn the_models_used_are_valid_and_big_enough_to_be_cut_into_blocks() {
    let c = config();
    c.validate().unwrap();
    let plan = ferrum_wgpu::llama::plan(&c, None);
    let biggest = plan
        .rows()
        .iter()
        .filter(|r| {
            !r.0.starts_with("scratch.")
                && (r.0.contains("gate") || r.0 == "head" || r.0 == "tok_emb")
        })
        .map(|r| r.1)
        .min()
        .unwrap();
    assert!(
        biggest > BLOCK_LIMIT,
        "every weight must exceed the block limit ({biggest})"
    );
}
