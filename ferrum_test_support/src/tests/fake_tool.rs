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

/// `dir/name` rendered the way the script text renders it (the platform's own separator).
fn at(dir: &Path, name: &str) -> String {
    dir.join(name).display().to_string()
}

#[test]
fn the_unix_script_logs_replies_complains_and_exits() {
    let d = Path::new("/d");
    let (log, reply, err) = (at(d, "log.txt"), at(d, "lms.reply0"), at(d, "lms.err"));
    assert_eq!(
        lms(d).unix_text(d, "lms"),
        format!(
            "#!/bin/sh\necho \"$@\" >> '{log}'\ncat \"$4\" >> '{log}'\n\
             if [ \"$1\" = 'ls' ]; then cat '{reply}'; fi\ncat '{err}' >&2\nexit 3\n"
        )
    );
    assert_eq!(FakeTool::default().unix_text(d, "t"), "#!/bin/sh\nexit 0\n");
}

#[test]
fn the_batch_file_does_the_same_with_crlf_lines() {
    let d = Path::new("C:\\d");
    let (log, reply, err) = (at(d, "log.txt"), at(d, "lms.reply0"), at(d, "lms.err"));
    let want = [
        "@echo off".to_string(),
        format!(">> \"{log}\" echo %*"),
        format!("type \"%~4\" >> \"{log}\""),
        format!("if \"%1\"==\"ls\" type \"{reply}\""),
        format!("type \"{err}\" 1>&2"),
        "exit /b 3".to_string(),
        String::new(),
    ];
    assert_eq!(lms(d).windows_text(d, "lms"), want.join("\r\n"));
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
