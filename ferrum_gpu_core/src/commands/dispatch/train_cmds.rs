use super::*;

/// `train`: a GPT-2 model (default) or `--arch llama`.
pub(super) fn train_cmd(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let a = Args::parse(
        &table(&[
            &DEVICE,
            &TRAIN,
            &gpt2::SIZE_FLAGS,
            &gpt2::TRAIN_FLAGS,
            &[val("--quant"), val("--arch"), val("--epochs")],
        ]),
        rest,
    )?;
    match a.get("--arch").unwrap_or("gpt2") {
        "gpt2" => gpt2::train(&a, out, None),
        "llama" => {
            if let Some(f) = gpt2::only_gpt2(&a) {
                return Err(format!("{f} needs --arch gpt2"));
            }
            train(&a, out, false)
        }
        o => Err(format!("--arch: unknown '{o}' (gpt2, llama)")),
    }
}

/// `finetune-gguf`: continue training a model file of either family.
pub(super) fn finetune_cmd(rest: &[String], out: &mut dyn Write) -> Result<(), String> {
    let a = Args::parse(
        &table(&[
            &DEVICE,
            &TRAIN,
            &gpt2::TRAIN_FLAGS,
            &[val("--model"), val("--quant"), val("--epochs")],
        ]),
        rest,
    )?;
    let model = a.need("--model")?;
    if gpt2::is_gpt2(model) {
        return gpt2::train(&a, out, Some(gpt2::load(model)?));
    }
    if let Some(f) = gpt2::only_gpt2(&a) {
        return Err(format!("{f} needs a gpt2 model"));
    }
    train(&a, out, true)
}
