use super::*;


/// Start the app: parse the flags, open the device, serve, open the window, and stop the server when
/// the app exits.
pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = match launch(&args) {
        Ok(Launch::Help(text)) => {
            println!("{text}");
            return;
        }
        Ok(Launch::Start(a)) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let server = match start_server(&args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot start: {e}");
            std::process::exit(1);
        }
    };
    // a closed stdout (a launcher that stopped reading) must not kill the app, as `println!` would
    let _ = writeln!(std::io::stdout(), "Ferrum GPU Studio: {}", server.url());
    configure(tauri::Builder::default(), server)
        .build(tauri::generate_context!())
        .expect("cannot build the Ferrum GPU Studio window")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                app.state::<Studio>().stop();
            }
        });
}
