use super::*;
mod data_and_export_cmds;
mod model_cmds;
mod train_cmds;
use data_and_export_cmds::*;
use model_cmds::*;
use train_cmds::*;

/// The command switch without the test-only lock, so one command (`run`) can invoke others: the
/// lock is not reentrant, and taking it twice on one thread deadlocks.
pub(super) fn dispatch(args: &[String], out: &mut dyn Write) -> Result<(), String> {
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    match cmd.as_str() {
        "-h" | "--help" | "help" => writeln!(out, "{USAGE}").map_err(|e| e.to_string()),
        "info" => info(
            &Args::parse(&table(&[&DEVICE, &[val("--model")]]), rest)?,
            out,
        ),
        "train" => train_cmd(rest, out),
        "lab" => lab::run(&Args::parse(&lab::FLAGS, rest)?, out),
        "run" => combined::run_combined(rest, out),
        "finetune-gguf" => finetune_cmd(rest, out),
        "generate" | "run-gguf" => generate_cmd(rest, out, cmd == "run-gguf"),
        "prepare" => prepare_cmd(rest, out),
        "tokenize" => tokenize_cmd(rest, out),
        "export-ollama" => export_ollama_cmd(rest, out),
        "export-lmstudio" => export_lmstudio_cmd(rest, out),
        "export-gguf" => export_gguf_cmd(rest, out),
        // eval-gguf is ferrum-cpu's name for scoring an imported model; ours scores any file
        "eval" | "eval-gguf" => eval_cmd(rest, out),
        other => Err(format!("unknown command '{other}'\n{USAGE}")),
    }
}
