use super::*;

pub(super) fn prepare_cmd(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let a = Args::parse(&gpt2::prepare::PREPARE_FLAGS, rest)?;
    gpt2::prepare::run(&a, rest, out)
}

pub(super) fn tokenize_cmd(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let flags = [
        val("--data"),
        val("--tokenizer"),
        val("--out"),
        val("--doc-sep"),
    ];
    gpt2::tokenize(&Args::parse(&flags, rest)?, out)
}

pub(super) fn export_ollama_cmd(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let flags = [
        val("--model"),
        val("--name"),
        val("--bin"),
        val("--modelfile-only"),
    ];
    runtimes::export_ollama(&Args::parse(&flags, rest)?, out)
}

pub(super) fn export_lmstudio_cmd(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let flags = [val("--model"), val("--repo"), val("--bin")];
    runtimes::export_lmstudio(&Args::parse(&flags, rest)?, out)
}

pub(super) fn export_gguf_cmd(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let flags = [val("--model"), val("--out"), val("--quant")];
    export_gguf(&Args::parse(&flags, rest)?, out)
}
