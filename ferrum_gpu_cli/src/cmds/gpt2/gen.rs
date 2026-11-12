//! `generate`, `run-gguf`, `eval` and `info` for GPT-2 files.

use super::score::mean_loss;
use crate::cmds::{app, device_line, mem_mb, say};
use crate::device::Workload;
use crate::flags::Args;
use ferrum_gguf::Gguf;
use ferrum_gpu_gui::app::App;
use ferrum_model::gpt2_gguf::from_gguf;
use ferrum_model::sampling::{sample, Params, Rng};
use ferrum_model::tokenizer_gguf;
use ferrum_ref::gpt2::Gpt2;
use ferrum_tok::ByteBpeTokenizer;
use std::io::Write;

/// A parsed model file: weights and tokenizer.
pub struct Loaded {
    /// The model.
    pub model: Gpt2,
    /// Its tokenizer (byte-level for a vocabulary of 256 without stored tokens).
    pub tok: ByteBpeTokenizer,
}

fn parse(path: &str) -> Result<Gguf, String> {
    Gguf::parse(std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)
        .map_err(|e| format!("{path}: {e}"))
}

/// Does `path` hold a `gpt2` model? (`false` for unreadable files; the real error comes later.)
pub fn is_gpt2(path: &str) -> bool {
    parse(path).is_ok_and(|g| g.string("general.architecture") == Ok("gpt2"))
}

/// Load a GPT-2 file and its tokenizer.
pub fn load(path: &str) -> Result<Loaded, String> {
    let g = parse(path)?;
    let model = from_gguf(&g).map_err(|e| format!("{path}: {e}"))?;
    let tok = if g.get("tokenizer.ggml.tokens").is_some() {
        tokenizer_gguf::from_gguf(&g).map_err(|e| format!("{path}: {e}"))?
    } else if model.cfg.vocab == 256 {
        ByteBpeTokenizer::byte_level()
    } else {
        return Err(format!(
            "{path}: the file has no tokenizer for its {} tokens",
            model.cfg.vocab
        ));
    };
    if tok.vocab_size() != model.cfg.vocab {
        return Err(format!(
            "{path}: the tokenizer has {} tokens but the model {}",
            tok.vocab_size(),
            model.cfg.vocab
        ));
    }
    Ok(Loaded { model, tok })
}

/// The `model:` line.
pub fn shape_of(path: &str) -> Result<String, String> {
    let c = load(path)?.model.cfg;
    Ok(format!(
        "model: gpt2 vocab {} ctx {} dim {} heads {} layers {} hidden {}",
        c.vocab, c.ctx, c.dim, c.heads, c.layers, c.hidden
    ))
}

/// Sampling and stopping options of one generation.
pub struct Opts {
    /// Most tokens to add.
    pub max_new: usize,
    /// Temperature, top-k, top-p, repetition penalty.
    pub params: Params,
    /// Sampler seed.
    pub seed: u64,
    /// Stop strings: the text ends before the first one found.
    pub stops: Vec<String>,
    /// Emit text as it is produced.
    pub stream: bool,
    /// Token ids that end the turn (end-of-sequence, `<|im_end|>`): sampling one stops, unprinted.
    pub stop_ids: Vec<usize>,
}

/// Parse the sampling and stopping flags of a generation.
pub fn opts(a: &Args) -> Result<Opts, String> {
    let name = |names: &[&'static str], d: &'static str| a.first(names).unwrap_or(d);
    let max_new = a
        .first(&["--max-new", "--max-tokens", "--chars"])
        .map_or(Ok(64usize), |n| a.num(n, 64))?;
    Ok(Opts {
        max_new,
        params: Params {
            temperature: a.num(name(&["--temperature", "--temp"], "--temperature"), 0.0f32)?,
            top_k: a.num("--top-k", 0)?,
            top_p: a.num("--top-p", 1.0f32)?,
            repetition_penalty: a.num("--repetition-penalty", 1.0f32)?,
        },
        seed: a.num(name(&["--seed", "--gen-seed"], "--seed"), 1u64)?,
        stops: a.all("--stop").to_vec(),
        stream: a.has("--stream"),
        stop_ids: vec![],
    })
}

/// Feed `prompt`, sample until `max_new`, a stop string, or a full context. Returns the new text
/// and the number of tokens sampled.
pub fn run(
    feed: &mut dyn FnMut(usize) -> Result<Vec<f32>, String>,
    decode: &dyn Fn(&[usize]) -> String,
    ctx: usize,
    prompt: &[usize],
    o: &Opts,
    emit: &mut dyn FnMut(&str),
) -> Result<(String, usize), String> {
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
        ferrum_gpu_gui::cancel::check()?;
        logits = feed(t)?;
    }
    let (mut history, mut new, mut shown) = (prompt.to_vec(), Vec::new(), 0usize);
    let mut text = String::new();
    while new.len() < o.max_new && history.len() < ctx {
        ferrum_gpu_gui::cancel::check()?;
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
            emit(&text[shown..safe]);
            shown = safe;
        }
        if history.len() < ctx && new.len() < o.max_new {
            logits = feed(next)?;
        }
    }
    if o.stream && text.len() > shown {
        emit(&text[shown..]);
    }
    Ok((text, new.len()))
}

/// A short greedy sample for `--sample`.
pub fn sample_text(
    app: &App,
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
        run(
            &mut *feed,
            &|ids| tok.decode(ids),
            m.cfg.ctx,
            &p,
            &o,
            &mut |_| {}
        )?
        .0
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
        &mut |s| {
            if let Err(e) = write!(out, "{s}") {
                werr.get_or_insert(e.to_string());
            }
        },
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
