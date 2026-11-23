use super::*;
use std::process::Command;

fn lms(dir: &Path) -> FakeTool {
    FakeTool {
        log: Some(dir.join("log.txt")),
        dump_arg: Some(4),
        replies: vec![("ls".into(), "[]".into())],
        stderr: "oops".into(),
        exit_code: 3,
    }
}

#[test]
fn the_unix_script_logs_replies_complains_and_exits() {
    let text = lms(Path::new("/d")).unix_text(Path::new("/d"), "lms");
    assert_eq!(
        text,
        "#!/bin/sh\necho \"$@\" >> '/d/log.txt'\ncat \"$4\" >> '/d/log.txt'\n\
         if [ \"$1\" = 'ls' ]; then cat '/d/lms.reply0'; fi\ncat '/d/lms.err' >&2\nexit 3\n"
    );
    assert_eq!(
        FakeTool::default().unix_text(Path::new("/d"), "t"),
        "#!/bin/sh\nexit 0\n"
    );
}

#[test]
fn the_batch_file_does_the_same_with_crlf_lines() {
    let text = lms(Path::new("C:\\d")).windows_text(Path::new("C:\\d"), "lms");
    let want = [
        "@echo off",
        ">> \"C:\\d/log.txt\" echo %*",
        "type \"%~4\" >> \"C:\\d/log.txt\"",
        "if \"%1\"==\"ls\" type \"C:\\d/lms.reply0\"",
        "type \"C:\\d/lms.err\" 1>&2",
        "exit /b 3",
        "",
    ];
    assert_eq!(text, want.join("\r\n"));
    assert_eq!(
        FakeTool::default().windows_text(Path::new("d"), "t"),
        "@echo off\r\nexit /b 0\r\n"
    );
}

#[test]
fn the_installed_program_runs_on_this_platform() {
    let dir = std::env::temp_dir().join(format!("ferrum_fake_tool_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let program = lms(&dir).install(&dir, "lms");
    let dumped = dir.join("dumped.txt");
    std::fs::write(&dumped, "file text").unwrap();
    let out = Command::new(&program)
        .args(["ls", "b", "c"])
        .arg(&dumped)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "[]");
    assert_eq!(String::from_utf8_lossy(&out.stderr).trim(), "oops");
    let log = std::fs::read_to_string(dir.join("log.txt")).unwrap();
    assert!(log.starts_with("ls b c "), "{log}");
    assert!(log.contains("file text"), "{log}");
}
