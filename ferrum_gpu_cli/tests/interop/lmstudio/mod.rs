//! LM Studio: `export-lmstudio` runs the right `lms` commands. The real `lms` registers models under
//! the user's `~/.lmstudio`, outside this repository, so a fake `lms` (a shell script that records
//! what it was asked and what file it was given) stands in; LM Studio's engine is llama.cpp, so that
//! the imported file runs is shown by loading the very bytes `lms` received in llama.cpp.
use crate::skip;
use crate::support::llama::completion;
use crate::support::model::{ferrum, fixture, ours};
use crate::support::tools::llamacpp_dir;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fake `lms` and the files it leaves behind.
struct Fake {
    bin: PathBuf,
    log: PathBuf,
    captured: PathBuf,
}

/// Write a fake `lms` into a fresh directory `tag`. On `import` it records the arguments and copies
/// the model file it was given; on `ls --json` it lists that file under `repo`. `import_exit` is the
/// status `import` returns.
fn fake(tag: &str, repo: &str, file: &str, import_exit: i32) -> Fake {
    let dir = fixture().dir.join(format!("lms-{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    let (log, captured) = (dir.join("calls.log"), dir.join("captured.gguf"));
    let script = format!(
        "#!/bin/sh\n\
         echo \"$@\" >> '{log}'\n\
         case \"$1\" in\n\
         import) for last; do :; done; cp \"$last\" '{cap}'; \
                 [ {exit} -eq 0 ] || {{ echo 'import refused by the fake lms' >&2; exit {exit}; }} ;;\n\
         ls) printf '[{{\"modelKey\":\"fake/{repo}\",\"path\":\"/home/u/.lmstudio/models/{repo}/{file}\"}}]\\n' ;;\n\
         esac\n",
        log = log.display(),
        cap = captured.display(),
        exit = import_exit,
    );
    let bin = dir.join("lms");
    std::fs::write(&bin, script).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    Fake { bin, log, captured }
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
