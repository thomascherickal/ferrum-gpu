use super::*;

/// The model a GGUF file holds, as text errors.
pub(super) fn from_bytes(bytes: Vec<u8>) -> Result<Llama, String> {
    let g = Gguf::parse(bytes).map_err(|e| e.to_string())?;
    from_gguf(&g).map_err(|e| e.to_string())
}

/// `eval` for a model with its own tokenizer: loss, perplexity and bits per byte of `--data`.
/// `--quant f32|int8|int4` scores the model after a round trip through that precision.
pub fn eval(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let mut l = load(a.need("--model")?)?;
    let text = std::fs::read_to_string(a.need("--data")?).map_err(|e| format!("--data: {e}"))?;
    let q = crate::commands::gpt2::quant(a, "f32")?;
    if a.get("--quant").is_some() {
        l.model = from_bytes(to_gguf(&l.model, q))?;
    }
    let max = l.model.cfg.ctx;
    let ctx = a.num("--ctx", 512usize.min(max))?;
    if ctx < 2 || ctx > max {
        return Err(format!("--ctx must be between 2 and {max} (got {ctx})"));
    }
    let mut ids: Vec<usize> = l.tok.prefix_bos().into_iter().collect();
    ids.extend(l.tok.encode(&text));
    let (loss, n) = mean_loss_ids(&l.model, &ids, ctx)?;
    let bpb = loss * n as f64 / std::f64::consts::LN_2 / text.len().max(1) as f64;
    say(
        out,
        format!(
            "tokens: {n} loss: {loss:.4} perplexity: {:.3} bits/byte: {bpb:.4}",
            loss.exp()
        ),
    )?;
    if let Some(p) = a.get("--json") {
        std::fs::write(
            p,
            format!(
                "{{\"tokens\":{n},\"bytes\":{},\"loss\":{loss},\"perplexity\":{},\"bits_per_byte\":{bpb}}}\n",
                text.len(),
                loss.exp()
            ),
        )
        .map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}
