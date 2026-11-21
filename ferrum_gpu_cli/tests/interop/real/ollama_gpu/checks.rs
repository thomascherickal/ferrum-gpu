use super::*;
use crate::support::ollama_server::Server;
use std::path::Path;

/// What the server chose, from its log.
pub(super) fn announce(server: &Server, mode: Mode) {
    step(format!(
        "{mode:?}: ready on {}; the server chose:",
        server.host
    ));
    let keys = [
        "inference compute",
        "library=",
        "discovered",
        "total vram",
        "GPU",
    ];
    for l in grep(&server.log_text(), &keys).iter().take(8) {
        step(format!("  ollama: {l}"));
    }
}

/// Export `model` to the server, generate raw and greedy, and demand the same text as ours with
/// part of the model in video memory.
pub(super) fn check_model(server: &Server, mode: Mode, tag: &str, label: &str, model: &Path) {
    let name = format!("ferrum-gpu-{tag}-{label}");
    step(format!(
        "{mode:?} {label}: ferrum-gpu export-ollama --model {} --name {name}",
        model.display()
    ));
    let o = server.export(model, &name);
    step(format!(
        "  export: {}",
        String::from_utf8_lossy(&o.stdout).trim()
    ));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    step(format!(
        "{mode:?} {label}: POST /api/generate (raw, greedy, GPU offload left to Ollama)"
    ));
    let theirs = server.raw_generate(&name, false);
    let mine = ours(model);
    step(format!("  ollama text: {:?}", format!("{PROMPT}{theirs}")));
    step(format!("  ferrum-gpu text: {mine:?}"));
    let (size, vram) = server
        .loaded_sizes(&name)
        .expect("the model is loaded (/api/ps)");
    step(format!(
        "  /api/ps: size {size} bytes, in video memory {vram} bytes"
    ));
    for l in grep(&server.log_text(), &["offloaded", "layers", "buffer size"])
        .iter()
        .rev()
        .take(4)
    {
        step(format!("  ollama: {l}"));
    }
    assert!(
        vram > 0,
        "{mode:?} {label}: Ollama kept the whole model off the GPU (size_vram 0)"
    );
    assert_eq!(
        format!("{PROMPT}{theirs}").trim_end(),
        mine,
        "{mode:?} {label}"
    );
    step(format!("  {mode:?} {label}: on the GPU and MATCH"));
}

/// The server log of the test's directory, for a failed start.
pub(super) fn server_log_tail(tag: &str) -> String {
    let p = fixture()
        .dir
        .join(format!("ollama-{tag}"))
        .join("server.log");
    let text = std::fs::read_to_string(p).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(25)..].join("\n")
}
