//! Ollama on the GPU, through each GPU backend it ships (CUDA and Vulkan). The proof that the GPU did
//! the work is Ollama's own accounting: `GET /api/ps` reports how many bytes of the loaded model sit
//! in video memory, and the server log names the library it picked.

use crate::skip;
use crate::support::log::{grep, opted_in, step};
use crate::support::model::{fixture, ours, PROMPT};
use crate::support::ollama_server::{start, Mode};
use crate::support::tools::ollama;

fn run(mode: Mode, tag: &str) {
    if !opted_in("FERRUM_INTEROP_GPU", "runs Ollama on the GPU") {
        return;
    }
    let Some(bin) = ollama() else {
        skip!("ollama not installed (set FERRUM_OLLAMA)")
    };
    step(format!(
        "{mode:?}: starting `ollama serve` ({}), private HOME and models, GPU allowed",
        bin.display()
    ));
    let Some(server) = start(&bin, tag, mode) else {
        step(format!(
            "{mode:?}: the server did not become ready; its log:\n{}",
            server_log_tail(tag)
        ));
        panic!("ollama serve ({mode:?}) did not start");
    };
    step(format!(
        "{mode:?}: ready on {}; the server chose:",
        server.host
    ));
    for l in grep(
        &server.log_text(),
        &[
            "inference compute",
            "library=",
            "discovered",
            "total vram",
            "GPU",
        ],
    )
    .iter()
    .take(8)
    {
        step(format!("  ollama: {l}"));
    }
    let f = fixture();
    for (label, model) in [("f32", &f.f32), ("q8_0", &f.q8)] {
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
}

/// The server log of the test's directory, for a failed start.
fn server_log_tail(tag: &str) -> String {
    let p = fixture()
        .dir
        .join(format!("ollama-{tag}"))
        .join("server.log");
    let text = std::fs::read_to_string(p).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(25)..].join("\n")
}

#[test]
fn ollama_on_the_gpu_through_cuda_gives_our_text() {
    run(Mode::Cuda, "gpu-cuda");
}

#[test]
fn ollama_on_the_gpu_through_vulkan_gives_our_text() {
    run(Mode::Vulkan, "gpu-vulkan");
}
