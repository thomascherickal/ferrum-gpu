use super::*;

#[test]
fn an_allocation_failure_while_uploading_blocks_is_reported() {
    let Some(g) = open_for_test() else { return };
    let table = vec![0.5f32; 17 * 32];
    g.fail_allocs_after(1);
    let e = Embedding::upload(&g, &table, 32, LIMIT).err().unwrap();
    assert!(e.to_string().contains("injected fault"), "{e}");
    let w = vec![0.5f32; 32 * 64];
    g.fail_allocs_after(1);
    let e = Lin::upload(&g, &w, (32, 64), None, LIMIT).err().unwrap();
    g.fail_allocs_after(usize::MAX);
    assert!(e.to_string().contains("injected fault"), "{e}");
}
