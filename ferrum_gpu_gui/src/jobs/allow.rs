//! What a job may run. The page cannot name a program or a shell string: it names one of a fixed
//! list of `ferrum-gpu` subcommands and a list of arguments, which are passed to the process as
//! separate arguments (never through a shell). The CLI itself rejects unknown and repeated flags;
//! this layer refuses what the CLI would accept but a web page should not be able to ask for.

use std::path::{Path, PathBuf};

/// The subcommands the Studio runs.
pub const COMMANDS: &[&str] = &[
    "info",
    "train",
    "finetune-gguf",
    "run",
    "generate",
    "run-gguf",
    "prepare",
    "tokenize",
    "export-ollama",
    "export-lmstudio",
    "export-gguf",
    "eval",
];

/// Most arguments in one request.
pub const MAX_ARGS: usize = 200;
/// Longest single argument, in bytes.
pub const MAX_ARG_BYTES: usize = 64 * 1024;

/// The only programs `--bin` may name (the CLI runs it): `ollama` and `lms`, with or without `.exe`.
fn allowed_tool(path: &str) -> bool {
    matches!(
        Path::new(path).file_name().and_then(|f| f.to_str()),
        Some("ollama" | "ollama.exe" | "lms" | "lms.exe")
    )
}

/// Check a job request; the error names the problem.
pub fn validate(cmd: &str, args: &[String]) -> Result<(), String> {
    if !COMMANDS.contains(&cmd) {
        return Err(format!("'{cmd}' is not a command the Studio runs"));
    }
    if args.len() > MAX_ARGS {
        return Err(format!("at most {MAX_ARGS} arguments"));
    }
    for (i, a) in args.iter().enumerate() {
        if a.len() > MAX_ARG_BYTES {
            return Err(format!("argument {i} is longer than {MAX_ARG_BYTES} bytes"));
        }
        if a.contains('\0') {
            return Err(format!("argument {i} contains a NUL byte"));
        }
        if i > 0 && args[i - 1] == "--bin" && !allowed_tool(a) {
            return Err("--bin may only name ollama or lms".into());
        }
    }
    if args.last().is_some_and(|a| a == "--bin") {
        return Err("--bin needs a value".into());
    }
    Ok(())
}

/// The `<out>.stop` file a training command watches, if the request names the file it writes (the
/// CLI uses `Path::new(target).with_extension("stop")`; `run` writes to `--model`).
pub fn stop_file(cmd: &str, args: &[String]) -> Option<PathBuf> {
    if !matches!(cmd, "train" | "finetune-gguf" | "run") {
        return None;
    }
    let flag = if cmd == "run" { "--model" } else { "--out" };
    let at = args.iter().position(|a| a == flag)?;
    Some(Path::new(args.get(at + 1)?).with_extension("stop"))
}
