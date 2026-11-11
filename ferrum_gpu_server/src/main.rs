//! `ferrum-gpu-server --model F.gguf [--port N] [--device cpu|gpu|auto] [--gpu-adapter NAME]`

use ferrum_gpu_cli::device::{decide, Workload};
use ferrum_gpu_cli::flags::{val, Args};
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_server::decoder::ModelDecoder;
use ferrum_gpu_server::serve::start;
use std::sync::Arc;

fn run(args: &[String]) -> Result<(), String> {
    let a = Args::parse(
        &[
            val("--model"),
            val("--port"),
            val("--device"),
            val("--gpu-adapter"),
            val("--threads"),
        ],
        args,
    )?;
    let model = ferrum_gpu_cli::cmds::load(a.need("--model")?)?;
    let choice = DeviceChoice::parse(a.get("--device").unwrap_or("auto"))?;
    let threads = a.num(
        "--threads",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )?;
    let d = decide(
        choice,
        Workload::Decode,
        &ferrum_wgpu::Gpu::list(),
        a.get("--gpu-adapter"),
    )?;
    let mut app = App::new(d.choice, d.adapter.as_deref(), threads.max(1))?;
    if d.note.is_some() {
        app.fallback_note = d.note;
    }
    println!("device: {} ({})", app.backend.name(), app.backend.kind());
    if let Some(n) = &app.fallback_note {
        println!("note: {n}");
    }
    let mem = a
        .get("--gpu-mem-mb")
        .map(|_| a.num("--gpu-mem-mb", 0u64))
        .transpose()?;
    ferrum_gpu_cli::cmds::gate(
        &mut std::io::stdout(),
        &app,
        &ferrum_gpu_cli::preflight::decode_buffers(&model),
        mem,
    )?;
    let server = start(
        Arc::new(ModelDecoder::new(app.backend.clone(), model)),
        a.num("--port", 0u16)?,
    )?;
    println!("ferrum-gpu server: http://{}/", server.addr);
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
