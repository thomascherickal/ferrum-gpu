//! `ferrum-gpu-server --model F.gguf [--port N] [--device cpu|gpu|auto] [--gpu-adapter NAME]`

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
    let app = App::new(choice, a.get("--gpu-adapter"), threads.max(1))?;
    println!("device: {} ({})", app.backend.name(), app.backend.kind());
    if let Some(n) = &app.fallback_note {
        println!("note: {n}");
    }
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
