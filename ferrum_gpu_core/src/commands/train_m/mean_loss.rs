use super::*;

/// Mean next-token loss of `text` under `m`, over non-overlapping windows of at most the context.
// ponytail: always on the CPU oracle; use GpuLlama::score for big models if eval gets slow.
pub fn mean_loss(m: &Llama, text: &[u8]) -> Result<(f64, usize), String> {
    let ids: Vec<usize> = text.iter().map(|&b| usize::from(b)).collect();
    mean_loss_ids(m, &ids, m.cfg.ctx)
}

/// Mean next-token loss of token `ids` over non-overlapping windows of at most `ctx` tokens.
pub fn mean_loss_ids(m: &Llama, ids: &[usize], ctx: usize) -> Result<(f64, usize), String> {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for w in ids.chunks(ctx.clamp(2, m.cfg.ctx)) {
        if w.len() < 2 {
            continue;
        }
        let logits = m.forward_logits(w)?;
        for i in 0..w.len() - 1 {
            let row = &logits[i * m.cfg.vocab..(i + 1) * m.cfg.vocab];
            let max = row.iter().copied().fold(f32::MIN, f32::max);
            let lse = row
                .iter()
                .map(|&x| f64::from(x - max).exp())
                .sum::<f64>()
                .ln();
            sum += lse + f64::from(max) - f64::from(row[w[i + 1]]);
            n += 1;
        }
    }
    if n == 0 {
        return Err("the data needs at least 2 tokens".into());
    }
    Ok((sum / n as f64, n))
}

pub(crate) fn eval(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let model = load(a.need("--model")?)?;
    let text = std::fs::read(a.need("--data")?).map_err(|e| format!("--data: {e}"))?;
    let (loss, n) = mean_loss(&model, &text)?;
    say(
        out,
        format!("tokens: {n} loss: {loss:.4} perplexity: {:.3}", loss.exp()),
    )
}
