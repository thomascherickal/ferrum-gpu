//! The jobs allow-list against shell-style injection: the command is a fixed name, never a path or
//! a shell string, and arguments are separate argv entries, so `;`, `--` and `$(...)` are inert text.
use super::support::{args, jobs_with, wait};
use crate::jobs::allow::validate;
use std::sync::{Arc, Mutex};

#[test]
fn command_names_that_smuggle_a_path_or_a_shell_are_refused() {
    for c in [
        "info;rm",
        "info ; ls",
        "info\n",
        "INFO",
        "/usr/bin/ferrum-gpu",
        "./info",
        "C:\\ferrum-gpu.exe",
        "info&&ls",
        "$(info)",
        "info --",
    ] {
        let e = validate(c, &[]).unwrap_err();
        assert!(e.contains("not a command"), "{c:?}: {e}");
    }
}

#[test]
fn shell_characters_and_double_dash_reach_the_command_as_plain_arguments() {
    let hostile = ["--", "; rm -rf /", "$(id)", "`id`", "a | b"];
    assert!(validate("eval", &hostile.map(String::from)).is_ok());
    let seen = Arc::new(Mutex::new(vec![]));
    let s2 = seen.clone();
    let jobs = jobs_with(move |argv, _| {
        s2.lock().unwrap().extend(argv.iter().cloned());
        Ok(())
    });
    let job = jobs.start("eval", args(&hostile)).unwrap();
    wait(&job, 20);
    let mut want = vec!["eval"];
    want.extend(hostile);
    assert_eq!(*seen.lock().unwrap(), want);
    // `--bin` after `--` is still judged: the CLI, not a shell, reads the list
    assert!(validate("export-ollama", &args(&["--", "--bin", "/bin/sh"])).is_err());
    // the inline form the CLI also accepts was not judged before: it named any program
    for bad in ["--bin=/bin/sh", "--bin=sh", "--bin="] {
        let e = validate("export-ollama", &args(&[bad])).unwrap_err();
        assert!(e.contains("ollama or lms"), "{bad}: {e}");
    }
    assert!(validate("export-ollama", &args(&["--bin=/opt/ollama"])).is_ok());
}
