use super::*;

/// The Modelfile text for a model at `abs` with `ctx` tokens of context. The path is
/// double-quoted, so callers reject paths with `"` or a newline first.
pub fn modelfile(abs: &Path, ctx: usize) -> String {
    format!(
        "FROM \"{}\"\nTEMPLATE \"{{{{ .Prompt }}}}\"\nPARAMETER stop \"<|endoftext|>\"\nPARAMETER temperature 0.8\nPARAMETER num_ctx {ctx}\n",
        abs.display()
    )
}

/// The absolute path of a GPT-2 model file and its context length; anything else is refused.
pub(super) fn runnable(model: &str) -> Result<(PathBuf, usize), String> {
    let g = ferrum_gguf::Gguf::parse(
        std::fs::read(model).map_err(|e| format!("cannot read GGUF {model}: {e}"))?,
    )
    .map_err(|e| format!("cannot read GGUF {model}: {e}"))?;
    let arch = g.string("general.architecture").unwrap_or("");
    if arch != "gpt2" {
        return Err(format!(
            "{model}: architecture '{arch}' cannot be exported; only gpt2 models (what `train` writes) run in Ollama and LM Studio from a Modelfile or `lms import`"
        ));
    }
    let ctx = g
        .uint("gpt2.context_length")
        .map_err(|e| format!("{model}: {e}"))?;
    let abs = std::path::absolute(model).map_err(|e| format!("{model}: {e}"))?;
    let s = abs.to_string_lossy();
    if s.contains('"') || s.contains('\n') {
        return Err(format!("{s}: the path contains a double quote or newline, which a Modelfile cannot quote; rename the file"));
    }
    Ok((abs, ctx))
}

/// Find a tool: `--bin`, then the environment variable, then `PATH`, then `fallback` under HOME.
pub fn find_tool(
    flag: Option<&str>,
    env: &str,
    name: &str,
    fallback: &str,
) -> Result<PathBuf, String> {
    find_tool_in(
        flag,
        std::env::var(env).ok().as_deref(),
        env,
        std::env::var_os("PATH").as_deref(),
        name,
        fallback,
    )
}
