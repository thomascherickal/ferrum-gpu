use super::*;

/// `generate` and `run-gguf` (`raw` is `run-gguf`): the family follows the model file.
pub(super) fn generate_cmd(rest: &[String], out: &mut dyn Write, raw: bool) -> Result<(), String> {
    let a = Args::parse(
        &table(&[
            &DEVICE,
            &GEN,
            &gpt2::GEN_FLAGS,
            &[val("--seed"), val("--max-new"), switch("--raw")],
        ]),
        rest,
    )?;
    if gpt2::is_gpt2(a.need("--model")?) {
        gpt2::generate(&a, out, raw)
    } else if llm::has_tokenizer(a.need("--model")?) {
        llm::generate(&a, out, raw)
    } else {
        generate(&a, out, raw)
    }
}

/// `eval` and `eval-gguf`: score any file.
pub(super) fn eval_cmd(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let flags = [
        val("--model"),
        val("--data"),
        val("--ctx"),
        val("--json"),
        val("--quant"),
    ];
    let a = Args::parse(&table(&[&flags]), rest)?;
    if gpt2::is_gpt2(a.need("--model")?) {
        gpt2::eval(&a, out)
    } else if llm::has_tokenizer(a.need("--model")?) {
        llm::eval(&a, out)
    } else {
        eval(&a, out)
    }
}
