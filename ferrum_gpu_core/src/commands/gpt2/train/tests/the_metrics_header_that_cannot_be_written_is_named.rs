use super::*;

#[test]
fn the_metrics_header_that_cannot_be_written_is_named() {
    // a zero-length slice accepts no bytes, so the header write fails like a full disk
    let e = with_header(&mut [0u8; 0][..], "m.csv")
        .map(|_| ())
        .unwrap_err();
    assert!(e.starts_with("m.csv: "), "{e}");
    let mut kept = Vec::new();
    with_header(&mut kept, "m.csv").unwrap();
    assert!(kept.starts_with(b"step,epoch,loss"));
}
