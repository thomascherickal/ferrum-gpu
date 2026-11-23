use super::*;

#[test]
fn a_tool_that_failed_gives_no_output() {
    assert_eq!(
        readers::stdout_if_ok(true, b"17179869184\n").as_deref(),
        Some("17179869184\n")
    );
    assert_eq!(readers::stdout_if_ok(false, b"partial"), None);
}
