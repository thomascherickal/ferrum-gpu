use super::*;

/// Largest training text accepted (bytes).
pub const MAX_TEXT: usize = 1 << 20;
/// Largest prompt accepted (bytes).
pub const MAX_PROMPT: usize = 4096;
/// Most tokens one generate call may add.
pub const MAX_NEW: usize = 1024;
/// Largest model (parameters) the GUI will hold; bigger uploads are refused.
pub const MAX_PARAMS: usize = 60_000_000;
/// Largest request body the server reads (a GGUF upload).
pub const MAX_BODY: usize = 256 << 20;

pub(crate) fn num(n: usize) -> Json {
    Json::Num(n as f64)
}

pub(crate) fn body(req: &Request) -> Result<Json, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| Response::error(400, "the body is not UTF-8"))?;
    Json::parse(text).map_err(|e| Response::error(400, &format!("bad JSON: {e}")))
}

/// An optional number field: absent gives `default`, wrong type or range is an error.
pub(crate) fn field(j: &Json, key: &str, default: f64, lo: f64, hi: f64) -> Result<f64, Response> {
    match j.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(v) => match v.as_f64() {
            Some(n) if n >= lo && n <= hi => Ok(n),
            _ => Err(Response::error(
                400,
                &format!("'{key}' must be a number in {lo}..={hi}"),
            )),
        },
    }
}

pub(super) fn model_json(m: &Llama) -> Json {
    let c = &m.cfg;
    Json::obj([
        (
            "preset",
            App::preset_of(m).map_or(Json::Null, |p| Json::Str(p.into())),
        ),
        ("params", num(m.num_params())),
        ("vocab", num(c.vocab)),
        ("ctx", num(c.ctx)),
        ("dim", num(c.dim)),
        ("layers", num(c.layers)),
        ("heads", num(c.heads)),
        ("kv_heads", num(c.kv_heads)),
        ("hidden", num(c.hidden)),
        (
            "rope",
            Json::Str(format!("{:?}", c.rope_type).to_lowercase()),
        ),
        ("tied", Json::Bool(m.tied)),
        ("byte_tokens", Json::Bool(c.vocab == tokenizer::VOCAB)),
    ])
}
