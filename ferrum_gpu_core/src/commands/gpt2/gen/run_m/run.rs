use super::*;

/// Feed `prompt`, sample until `max_new`, a stop string, or a full context. Returns the new text
/// and the number of tokens sampled.
pub fn run(
    feed: &mut dyn FnMut(usize) -> Result<Vec<f32>, String>,
    decode: &dyn Fn(&[usize]) -> String,
    ctx: usize,
    prompt: &[usize],
    o: &Opts,
    mut emit: Option<&mut dyn FnMut(&str)>,
) -> Result<(String, usize), String> {
    let mut send = |s: &str| {
        if let Some(e) = emit.as_mut() {
            e(s);
        }
    };
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    if prompt.len() >= ctx {
        return Err(format!(
            "the prompt has {} tokens but the context is {ctx}",
            prompt.len()
        ));
    }
    let mut rng = Rng::new(o.seed);
    let mut logits = Vec::new();
    for &t in prompt {
        crate::cancel::check()?;
        logits = feed(t)?;
    }
    let (mut history, mut new, mut shown) = (prompt.to_vec(), Vec::new(), 0usize);
    let mut text = String::new();
    while new.len() < o.max_new && history.len() < ctx {
        crate::cancel::check()?;
        let next = sample(&logits, &o.params, &history, &mut rng);
        if o.stop_ids.contains(&next) {
            break;
        }
        new.push(next);
        history.push(next);
        text = decode(&new);
        if let Some(at) = o
            .stops
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| text.find(s.as_str()))
            .min()
        {
            text.truncate(at);
            break;
        }
        // hold back a trailing replacement character: the next token may complete it
        let safe = if text.ends_with('\u{fffd}') {
            text.len() - 3
        } else {
            text.len()
        };
        if o.stream && safe > shown {
            send(&text[shown..safe]);
            shown = safe;
        }
        if history.len() < ctx && new.len() < o.max_new {
            logits = feed(next)?;
        }
    }
    if o.stream && text.len() > shown {
        send(&text[shown..]);
    }
    Ok((text, new.len()))
}
