use super::*;

#[test]
fn a_fault_at_any_readback_fails_training_cleanly() {
    let Some(g) = open_for_test() else { return };
    let g = Arc::new(g);
    let (b, llama) = (GpuBackend::new(g.clone()), fresh_model("tiny", 7).unwrap());
    let reads = (0..100)
        .find(|&k| {
            g.fail_reads_after(k);
            match b.train(&llama, &spec(1), &shared()) {
                Ok(_) => true,
                Err(e) => {
                    assert!(e.contains("injected fault"), "read {k}: {e}");
                    false
                }
            }
        })
        .expect("training stops reading within 100 readbacks");
    g.fail_reads_after(usize::MAX);
    assert!(
        reads > 1,
        "a step reads its loss and the weights are read at the end"
    );
}
