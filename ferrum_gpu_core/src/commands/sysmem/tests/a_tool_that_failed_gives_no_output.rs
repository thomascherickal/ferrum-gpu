use super::*;
use std::os::unix::process::ExitStatusExt;
use std::process::{ExitStatus, Output};

fn exited(code: i32, stdout: &str) -> Output {
    Output {
        status: ExitStatus::from_raw(code << 8),
        stdout: stdout.as_bytes().to_vec(),
        stderr: vec![],
    }
}

#[test]
fn a_tool_that_failed_gives_no_output() {
    assert_eq!(
        readers::stdout_if_ok(&exited(0, "17179869184\n")).as_deref(),
        Some("17179869184\n")
    );
    assert_eq!(readers::stdout_if_ok(&exited(1, "partial")), None);
}
