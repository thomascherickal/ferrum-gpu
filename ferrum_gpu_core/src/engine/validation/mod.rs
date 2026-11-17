//! A validation split for the GUI's byte-level training: hold out the end of the text, then
//! measure the loss on it before and after the run.

use ferrum_ref::llama::Llama;

/// The largest share of the text that may be held out.
pub const MAX_FRACTION: f32 = 0.5;

/// Fewest bytes a validation share may be (a loss needs one next-token pair).
pub const MIN_BYTES: usize = 2;

/// Split `text` into `(train, validation)`: the last `fraction` of the bytes, at least
/// [`MIN_BYTES`], is held out. A fraction of `0` holds nothing out.
pub fn split(text: &[u8], fraction: f32) -> Result<(&[u8], &[u8]), String> {
    if fraction == 0.0 {
        return Ok((text, &[]));
    }
    if !(fraction > 0.0 && fraction <= MAX_FRACTION) {
        return Err(format!(
            "the validation share must be in (0, {MAX_FRACTION}], got {fraction}"
        ));
    }
    let held = ((text.len() as f64 * f64::from(fraction)) as usize).max(MIN_BYTES);
    if held >= text.len() {
        return Err(format!(
            "the text has {} bytes, too few to hold out a validation share",
            text.len()
        ));
    }
    Ok(text.split_at(text.len() - held))
}

/// Mean next-token cross-entropy of `m` on `text` (bytes are the token ids) over non-overlapping
/// windows of at most `context` tokens; `None` when the text has no next-token pair or the model
/// cannot score it.
pub fn mean_loss(m: &Llama, text: &[u8], context: usize) -> Option<f32> {
    let (mut sum, mut n) = (0.0f64, 0usize);
    let ids: Vec<usize> = text.iter().map(|&b| usize::from(b)).collect();
    for w in ids.chunks(context.clamp(2, m.cfg.ctx)) {
        if w.len() < 2 {
            continue;
        }
        let logits = m.forward_logits(w).ok()?;
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
    (n > 0).then(|| (sum / n as f64) as f32)
}

#[cfg(test)]
mod tests;
