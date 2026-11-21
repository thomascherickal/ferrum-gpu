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

/// A short greedy sample for `--sample`.
pub fn sample_text(
    app: &Session,
    m: &Gpt2,
    tok: &ByteBpeTokenizer,
    prompt: &str,
) -> Result<String, String> {
    let mut feed = app.backend.gpt2_decoder(m)?;
    let o = Opts {
        max_new: 32,
        params: Params::default(),
        seed: 1,
        stops: vec![],
        stream: false,
        stop_ids: vec![],
    };
    let p = tok.encode(prompt);
    Ok(format!(
        "{prompt}{}",
        run(&mut *feed, &|ids| tok.decode(ids), m.cfg.ctx, &p, &o, None)?.0
    ))
}

/// `generate` / `run-gguf` for a GPT-2 file.
pub fn generate(a: &Args, out: &mut dyn Write, verbose: bool) -> Result<(), String> {
    let path = a.need("--model")?;
    let l = load(path)?;
    let o = opts(a)?;
    let prompt = match a.get("--ids") {
        Some(s) => s
            .split_whitespace()
            .map(|x| {
                x.parse::<usize>()
                    .map_err(|_| format!("--ids: '{x}' is not an id"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => l.tok.encode(a.need("--prompt")?),
    };
    let app = app(a, Workload::Decode)?;
    if verbose {
        device_line(out, &app)?;
        say(out, shape_of(path)?)?;
    }
    let _ = mem_mb(a)?;
    let mut feed = app.backend.gpt2_decoder(&l.model)?;
    let shown = a.get("--prompt").unwrap_or("");
    if o.stream {
        write!(out, "{shown}").map_err(|e| e.to_string())?;
    }
    let mut werr = None;
    let (text, _) = run(
        &mut *feed,
        &|ids| l.tok.decode(ids),
        l.model.cfg.ctx,
        &prompt,
        &o,
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
