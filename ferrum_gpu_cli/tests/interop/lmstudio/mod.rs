//! LM Studio: `export-lmstudio` runs the right `lms` commands. The real `lms` registers models under
//! the user's `~/.lmstudio`, outside this repository, so a fake `lms` (a script that records
//! what it was asked) stands in; LM Studio's engine is llama.cpp, so that
//! the imported file runs is shown by loading the very bytes `lms` received in llama.cpp.
use crate::skip;
use crate::support::llama::completion;
use crate::support::model::{ferrum, fixture, ours};
use crate::support::tools::llamacpp_dir;
use ferrum_test_support::FakeTool;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fake `lms` and the file of the calls it logged.
struct Fake {
    bin: PathBuf,
    log: PathBuf,
}

/// Write a fake `lms` into a fresh directory `tag`, from one description that becomes a shell
/// script on Unix and a batch file on Windows. It logs every call's arguments; on `ls --json` it
/// lists `file` under `repo`; `import_exit` is the status it returns (with a message on stderr).
fn fake(tag: &str, repo: &str, file: &str, import_exit: i32) -> Fake {
    let dir = fixture().dir.join(format!("lms-{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("calls.log");
    let listing = format!(
        "[{{\"modelKey\":\"fake/{repo}\",\"path\":\"/home/u/.lmstudio/models/{repo}/{file}\"}}]"
    );
    let tool = FakeTool {
        log: Some(log.clone()),
        replies: vec![("ls".into(), listing)],
        stderr: if import_exit == 0 {
            String::new()
        } else {
            "import refused by the fake lms".into()
        },
        exit_code: import_exit,
        ..FakeTool::default()
    };
    Fake {
        bin: tool.install(&dir, "lms"),
        log,
    }
}

/// The calls `lms` logged, one per line, without the quotes `cmd.exe` keeps around arguments.
fn logged_calls(fk: &Fake) -> Vec<String> {
    let log = std::fs::read_to_string(&fk.log).unwrap();
    log.lines().map(|l| l.replace('"', "")).collect()
}

/// The path `lms import` was given, from the first logged call.
fn imported_path(fk: &Fake) -> PathBuf {
    let calls = logged_calls(fk);
    let (_, rest) = calls[0].rsplit_once(" --user-repo ").unwrap();
    PathBuf::from(rest.split_once(' ').unwrap().1)
}

fn export(extra: &[&str], model: &Path, tool: &Path) -> Output {
    let mut args = vec![
        "export-lmstudio",
        "--model",
        model.to_str().unwrap(),
        "--bin",
        tool.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    ferrum(&args)
}

mod a_failing_import_reports_what_lms_said;
mod it_imports_with_the_lms_command_line_lm_studio_expects;
