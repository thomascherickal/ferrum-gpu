use super::*;

/// Single-turn prompt in `fmt`, ending where the assistant should speak.
pub fn render(fmt: ChatFormat, user: &str) -> Vec<Segment> {
    use Segment::*;
    match fmt {
        ChatFormat::ChatMl => vec![
            Special("<|im_start|>"),
            Text(format!("user\n{user}")),
            Special("<|im_end|>"),
            Text("\n".into()),
            Special("<|im_start|>"),
            Text("assistant\n".into()),
        ],
        ChatFormat::Llama3 => vec![
            Special("<|start_header_id|>"),
            Text("user".into()),
            Special("<|end_header_id|>"),
            Text(format!("\n\n{user}")),
            Special("<|eot_id|>"),
            Special("<|start_header_id|>"),
            Text("assistant".into()),
            Special("<|end_header_id|>"),
            Text("\n\n".into()),
        ],
        ChatFormat::Llama2Inst => vec![Text(format!("[INST] {user} [/INST]"))],
        ChatFormat::Zephyr => vec![
            Special("<|user|>"),
            Text(format!("\n{user}")),
            Special("</s>"),
            Text("\n".into()),
            Special("<|assistant|>"),
            Text("\n".into()),
        ],
    }
}

/// The token that ends the assistant turn in `fmt`.
pub fn stop_token(fmt: ChatFormat) -> &'static str {
    match fmt {
        ChatFormat::ChatMl => "<|im_end|>",
        ChatFormat::Llama3 => "<|eot_id|>",
        ChatFormat::Llama2Inst | ChatFormat::Zephyr => "</s>",
    }
}

/// Encode segments: specials map to their atomic vocab id when present
/// (falling back to plain text encoding — degraded but functional).
pub fn encode_segments(tok: &GgufTokenizer, segs: &[Segment]) -> Vec<usize> {
    let mut ids = Vec::new();
    for s in segs {
        match s {
            Segment::Special(p) => match tok.token_id(p) {
                Some(id) => ids.push(id),
                None => ids.extend(tok.encode(p)),
            },
            Segment::Text(t) => ids.extend(tok.encode(t)),
        }
    }
    ids
}
