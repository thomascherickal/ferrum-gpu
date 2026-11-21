//! Ollama on the GPU, through each GPU backend it ships (CUDA and Vulkan). The proof that the GPU did
//! the work is Ollama's own accounting: `GET /api/ps` reports how many bytes of the loaded model sit
//! in video memory, and the server log names the library it picked.
use crate::skip;
use crate::support::log::{grep, opted_in, step};
use crate::support::model::{fixture, ours, PROMPT};
use crate::support::ollama_server::{start, Mode};
use crate::support::tools::ollama;

mod checks;
use checks::*;

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
    announce(&server, mode);
    let f = fixture();
    for (label, model) in [("f32", &f.f32), ("q8_0", &f.q8)] {
        check_model(&server, mode, tag, label, model);
    }
}

mod ollama_on_the_gpu_through_cuda_gives_our_text;
