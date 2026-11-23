use super::*;
use crate::{GGML_F16, GGML_F32, GGML_I16, GGML_I32, GGML_Q8_0, GGML_Q8_1};

#[test]
fn absurd_element_counts_overflow_the_byte_size() {
    let huge = usize::MAX - usize::MAX % 32;
    for ty in [GGML_F32, GGML_F16, GGML_I16, GGML_I32, GGML_Q8_0, GGML_Q8_1] {
        let n = if ty == GGML_Q8_0 || ty == GGML_Q8_1 {
            huge
        } else {
            usize::MAX
        };
        assert_eq!(
            nbytes(ty, n).unwrap_err().0,
            "tensor size overflows",
            "type {ty}"
        );
    }
}
