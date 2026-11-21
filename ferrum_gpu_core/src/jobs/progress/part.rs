use super::*;

impl Event {
    /// The JSON the page reads.
    pub fn json(&self) -> Json {
        let n = |v: f64| Json::Num(v);
        match self {
            Event::Step {
                step,
                total,
                loss,
                grad_norm,
                tokens_per_s,
                elapsed,
                eta,
            } => Json::obj([
                ("kind", Json::Str("step".into())),
                ("step", n(*step as f64)),
                ("total", n(*total as f64)),
                ("loss", n(*loss)),
                ("grad_norm", n(*grad_norm)),
                ("tokens_per_s", n(*tokens_per_s)),
                ("elapsed", n(*elapsed)),
                ("eta", n(*eta)),
            ]),
            Event::Eval {
                tokens,
                loss,
                perplexity,
                bits_per_byte,
            } => Json::obj([
                ("kind", Json::Str("eval".into())),
                ("tokens", n(*tokens as f64)),
                ("loss", n(*loss)),
                ("perplexity", n(*perplexity)),
                ("bits_per_byte", bits_per_byte.map_or(Json::Null, n)),
            ]),
            Event::Val { epoch, loss, best } => Json::obj([
                ("kind", Json::Str("val".into())),
                ("epoch", n(*epoch as f64)),
                ("loss", n(*loss)),
                ("best", Json::Bool(*best)),
            ]),
        }
    }
}
