use super::*;

#[test]
fn the_batch_file_logs_without_dumping_an_argument() {
    let tool = FakeTool {
        log: Some("log.txt".into()),
        ..FakeTool::default()
    };
    assert_eq!(
        tool.windows_text(Path::new("d"), "t"),
        "@echo off\r\n>> \"log.txt\" echo %*\r\nexit /b 0\r\n"
    );
}
