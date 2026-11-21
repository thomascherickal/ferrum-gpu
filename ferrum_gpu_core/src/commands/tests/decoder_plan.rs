use super::fixtures::*;
use super::*;
use crate::engine::CpuBackend;

#[test]
fn the_decoder_plan_is_split_at_the_chosen_adapters_binding_limit() {
    let m = fresh_model("tiny", 1).unwrap();
    let whole = preflight::decode_buffers(&m);
    let biggest = whole
        .rows()
        .iter()
        .filter(|r| r.0.contains("blk0.gate") || r.0 == "head" || r.0 == "tok_emb")
        .map(|r| r.1)
        .max()
        .unwrap();
    // a limit between the unsplittable buffers and the biggest weight
    let unsplittable = whole
        .rows()
        .iter()
        .filter(|r| r.0.contains("cache") || r.0.starts_with("scratch"))
        .map(|r| r.1)
        .max()
        .unwrap();
    let limit = (biggest / 2).max(unsplittable);
    assert!(
        limit < biggest,
        "the tiny preset must have a weight bigger than its own KV cache ({limit} vs {biggest})"
    );
    let mut one = listed();
    one[0].max_binding = limit;
    let split = preflight::decode_buffers_for(&m, &session_on(Arc::new(FakeGpu), one));
    assert_eq!(split.total(), whole.total(), "splitting keeps the bytes");
    assert!(split.rows().len() > whole.rows().len());
    assert!(split.rows().iter().all(|r| r.1 <= limit));
    // a CPU backend has no limit, so the plan is the whole one
    let cpu =
        preflight::decode_buffers_for(&m, &session_on(Arc::new(CpuBackend::new(1)), listed()));
    assert_eq!(cpu.rows(), whole.rows());
}
