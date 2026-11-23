use super::*;

#[test]
#[should_panic(expected = "internal shape error")]
fn a_shape_the_matmul_refuses_is_an_internal_error() {
    mm(&[1.0], &[1.0], (2, 2, 2), 1);
}
