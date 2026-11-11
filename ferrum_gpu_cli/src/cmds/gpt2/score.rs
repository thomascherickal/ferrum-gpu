//! Mean next-token loss of a token stream under a GPT-2 model (the CPU oracle).

use ferrum_ref::gpt2::Gpt2;

/// Mean cross-entropy of `ids` over non-overlapping windows of at most `ctx` tokens, and the
/// number of predicted tokens. Needs at least two tokens.
pub fn mean_loss(m: &Gpt2, ids: &[usize], ctx: usize) -> Result<(f64, usize), String> {
    let (mut sum, mut n) = (0.0f64, 0usize);
    let v = m.cfg.vocab;
    for w in ids.chunks(ctx.clamp(2, m.cfg.ctx)) {
        if w.len() < 2 {
            continue;
        }
        let logits = m.forward_logits(w)?;
        for i in 0..w.len() - 1 {
            let row = &logits[i * v..(i + 1) * v];
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
