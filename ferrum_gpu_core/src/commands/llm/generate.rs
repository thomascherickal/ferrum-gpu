use super::*;

/// `generate` / `run-gguf` for a model with its own tokenizer.
pub fn generate(a: &Args, out: &mut dyn Write, verbose: bool) -> Result<(), String> {
    let l = load(a.need("--model")?)?;
    let mut o = opts(a)?;
    let (prompt, chat) = prompt_ids(&l, a)?;
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    o.stop_ids = l.tok.eos().into_iter().collect();
    if let (true, Some(f)) = (chat, l.chat) {
        o.stop_ids.extend(l.tok.token_id(stop_token(f)));
    }
    let app = app(a, Workload::Decode)?;
    if verbose {
        device_line(out, &app)?;
        say(out, shape(&l.model))?;
    }
    gate(
        out,
        &app,
        &preflight::decode_buffers_for(&l.model, &app),
        mem_mb(a)?,
    )?;
    let mut feed = app.backend.llama_decoder(&l.model)?;
    let anchor = prompt[prompt.len() - 1];
    let shown = if chat {
        String::new()
    } else {
        a.get("--prompt").unwrap_or("").to_string()
    };
    if o.stream {
        write!(out, "{shown}").map_err(|e| e.to_string())?;
    }
    let mut werr = None;
    let (text, _) = run(
        &mut *feed,
        &|ids| continuation(&l.tok, anchor, ids),
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
