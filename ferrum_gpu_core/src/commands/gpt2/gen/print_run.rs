use super::*;

/// Next-token logits for a fed token (the decoder), and ids to text (the tokenizer).
pub type DecodeFns<'a> = (
    &'a mut dyn FnMut(usize) -> Result<Vec<f32>, String>,
    &'a dyn Fn(&[usize]) -> String,
);

/// [`run`] the generation and print it to `out` after `shown` (the visible prompt): token by
/// token when `o.stream`, else all at once. A failed write while streaming is reported after
/// the run.
pub fn print_run(
    out: &mut dyn Write,
    (feed, decode): DecodeFns<'_>,
    (ctx, prompt): (usize, &[usize]),
    o: &Opts,
    shown: &str,
) -> Result<(), String> {
    if o.stream {
        write!(out, "{shown}").map_err(|e| e.to_string())?;
    }
    let mut werr = None;
    let (text, _) = run(
        feed,
        decode,
        ctx,
        prompt,
        o,
        Some(&mut |s| {
            if let Err(e) = write!(out, "{s}") {
                werr.get_or_insert(e.to_string());
            }
        }),
    )?;
    if let Some(e) = werr {
        return Err(e);
    }
    if o.stream {
        writeln!(out).map_err(|e| e.to_string())
    } else {
        say(out, format!("{shown}{text}"))
    }
}
