use super::*;

#[test]
fn a_codec_error_becomes_a_gpu_error() {
    // 3 bytes cannot be 32 Q8_0 elements (one block is 34 bytes).
    let e = dequantise_bytes(QKind::Q8_0, &[0; 3], 32).unwrap_err();
    assert!(e.to_string().contains("does not match"), "{e}");
}
