//! `ferrum-gpu`: the command line. See `ferrum-gpu help`. The commands live in `ferrum_gpu_core`
//! (`commands`), where the Studio runs the same functions in-process.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = ferrum_gpu_core::commands::run(&args, &mut std::io::stdout()) {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
