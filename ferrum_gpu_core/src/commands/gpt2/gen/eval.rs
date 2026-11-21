use super::*;

/// `eval` for a GPT-2 file: mean loss and perplexity of `--data`.
pub fn eval(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let l = load(a.need("--model")?)?;
    let text = std::fs::read_to_string(a.need("--data")?).map_err(|e| format!("--data: {e}"))?;
    let ctx = a.num("--ctx", l.model.cfg.ctx)?;
    let (loss, n) = mean_loss(&l.model, &l.tok.encode_docs(&text), ctx)?;
    say(
        out,
        format!("tokens: {n} loss: {loss:.4} perplexity: {:.3}", loss.exp()),
    )?;
    if let Some(p) = a.get("--json") {
        std::fs::write(
            p,
            format!(
                "{{\"tokens\":{n},\"loss\":{loss},\"perplexity\":{}}}\n",
                loss.exp()
            ),
        )
        .map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}
