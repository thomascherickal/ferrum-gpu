use super::*;

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
