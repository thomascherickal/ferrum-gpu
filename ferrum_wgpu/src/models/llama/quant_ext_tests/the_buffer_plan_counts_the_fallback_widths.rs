use super::*;

#[test]
fn the_buffer_plan_counts_the_fallback_widths() {
    let c = cfg(256, 704, 64);
    let whole = plan(&c, Some(QKind::Q2K)).total();
    let q4 = plan(&c, Some(QKind::Q4_0)).total();
    let f32_total = plan(&c, None).total();
    // smaller than f32, and larger than all-Q4_0 would be only if a matrix kept more than 4 bits
    assert!(whole < f32_total && whole < q4, "{whole} {q4} {f32_total}");
}
