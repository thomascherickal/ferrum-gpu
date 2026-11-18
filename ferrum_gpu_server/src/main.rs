//! `ferrum-gpu-server --model F.gguf [--host H] [--port N] [--token T] [--device cpu|gpu|auto] ...`
//! Serves GPT-2 files (with their BPE tokenizer) and Llama-style byte-level files. The server itself
//! lives in `ferrum_gpu_core::serving`, where the Studio starts the same one in-process.

use ferrum_gpu_core::serving::boot::{boot, wants_help, USAGE};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if wants_help(&args) {
        // a closed stdout (`ferrum-gpu-server | head -1`) must not panic, which `println!` would
        let _ = writeln!(std::io::stdout(), "{USAGE}");
        return;
    }
    match boot(&args, &mut std::io::stdout()) {
        Ok(_running) => ferrum_gpu_core::hold::hold(),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}
