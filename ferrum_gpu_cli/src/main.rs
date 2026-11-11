//! `ferrum-gpu`: the command line. See `ferrum-gpu help`.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = ferrum_gpu_cli::cmds::run(&args, &mut std::io::stdout()) {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
