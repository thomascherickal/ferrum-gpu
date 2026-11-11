//! `ferrum-gpu-server --model F.gguf [--host H] [--port N] [--token T] [--device cpu|gpu|auto] ...`
//! Serves GPT-2 files (with their BPE tokenizer) and Llama-style byte-level files.

use ferrum_gpu_cli::cmds::gpt2;
use ferrum_gpu_cli::device::{decide, Workload};
use ferrum_gpu_cli::flags::{switch, val, Args};
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_server::decoder::{Decoder, Gpt2Decoder, ModelDecoder};
use ferrum_gpu_server::limits::Limits;
use ferrum_gpu_server::serve::{check_exposure, start, Config};
use std::sync::Arc;

const USAGE: &str = "usage: ferrum-gpu-server --model F.gguf [--host 127.0.0.1] [--port 8080] [--token T | FERRUM_TOKEN] \
[--allow-unauthenticated] [--max-batch N] [--max-chars N] [--max-body BYTES] [--max-tokens N] \
[--repetition-penalty X] [--device cpu|gpu|auto] [--gpu-adapter NAME] [--gpu-mem-mb N] [--threads N]";

fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let a = Args::parse(
        &[
            val("--model"),
            val("--host"),
            val("--port"),
            val("--token"),
            switch("--allow-unauthenticated"),
            switch("--force"),
            val("--max-batch"),
            val("--max-chars"),
            val("--max-body"),
            val("--max-tokens"),
            val("--repetition-penalty"),
            val("--device"),
            val("--gpu-adapter"),
            val("--gpu-mem-mb"),
            val("--threads"),
        ],
        args,
    )?;
    let d = Limits::default();
    let token = a
        .get("--token")
        .map(String::from)
        .or_else(|| std::env::var("FERRUM_TOKEN").ok().filter(|t| !t.is_empty()));
    let cfg = Config {
        host: a.get("--host").unwrap_or("127.0.0.1").to_string(),
        port: a.num("--port", 8080u16)?,
        token,
        limits: Limits {
            max_batch: a.num("--max-batch", d.max_batch)?,
            max_chars: a.num("--max-chars", d.max_chars)?,
            max_body: a.num("--max-body", d.max_body)?,
            max_tokens: a.num("--max-tokens", d.max_tokens)?,
            repetition_penalty: a.num("--repetition-penalty", d.repetition_penalty)?,
        },
    };
    check_exposure(
        &cfg.host,
        cfg.token.as_deref(),
        a.has("--allow-unauthenticated"),
    )?;
    let path = a.need("--model")?;
    let choice = DeviceChoice::parse(a.get("--device").unwrap_or("auto"))?;
    let threads = a.num(
        "--threads",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )?;
    let dec = decide(
        choice,
        Workload::Decode,
        &ferrum_wgpu::Gpu::list(),
        a.get("--gpu-adapter"),
    )?;
    let mut app = App::new(dec.choice, dec.adapter.as_deref(), threads.max(1))?;
    if dec.note.is_some() {
        app.fallback_note = dec.note;
    }
    println!("device: {} ({})", app.backend.name(), app.backend.kind());
    if let Some(n) = &app.fallback_note {
        println!("note: {n}");
    }
    let mem = a
        .get("--gpu-mem-mb")
        .map(|_| a.num("--gpu-mem-mb", 0u64))
        .transpose()?;
    let decoder: Arc<dyn Decoder> = if gpt2::is_gpt2(path) {
        let l = gpt2::load(path)?;
        ferrum_gpu_cli::cmds::gate(
            &mut std::io::stdout(),
            &app,
            &ferrum_gpu_cli::preflight::gpt2_decode_buffers(&l.model),
            mem,
        )?;
        Arc::new(Gpt2Decoder::new(app.backend.clone(), l))
    } else {
        let model = ferrum_gpu_cli::cmds::load(path)?;
        ferrum_gpu_cli::cmds::gate(
            &mut std::io::stdout(),
            &app,
            &ferrum_gpu_cli::preflight::decode_buffers(&model),
            mem,
        )?;
        Arc::new(ModelDecoder::new(app.backend.clone(), model))
    };
    let server = start(decoder, &cfg)?;
    println!("ferrum-gpu server: http://{}/", server.addr);
    if cfg.token.is_some() {
        println!("auth: bearer token required (GET /health is open)");
    }
    loop {
        std::thread::park();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
