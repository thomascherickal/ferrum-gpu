use crate::common::console_lines;

#[test]
fn console_and_error_lines_are_picked_and_a_quiet_log_shows_its_tail() {
    let log = "boot\n[1:2] CONSOLE hello\nnoise\n[1:2] ERROR: bad\n";
    assert_eq!(console_lines(log), "[1:2] CONSOLE hello\n[1:2] ERROR: bad");
    let quiet = (1..=20)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let shown = console_lines(&quiet);
    assert!(
        shown.starts_with("(no console lines; 20 log lines, tail)\nline 6\n"),
        "{shown}"
    );
    assert!(shown.ends_with("line 20"), "{shown}");
}
