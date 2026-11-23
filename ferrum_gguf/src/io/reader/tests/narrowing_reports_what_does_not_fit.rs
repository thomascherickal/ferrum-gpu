use super::*;

#[test]
fn narrowing_reports_what_does_not_fit() {
    assert_eq!(narrow::<u8>(200, "x overflows").unwrap(), 200);
    assert_eq!(
        narrow::<u8>(256, "x overflows").unwrap_err().0,
        "x overflows"
    );
}
