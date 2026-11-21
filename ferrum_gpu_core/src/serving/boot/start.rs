use super::config::{parse_flags, server_config};
use super::decoder::{load_decoder, open_session};
use super::*;

/// Start the server for `args` (the flags of `ferrum-gpu-server`), writing its status lines to `out`.
pub fn boot(args: &[String], out: &mut dyn Write) -> Result<Booted, String> {
    let a = parse_flags(args)?;
    let cfg = server_config(&a)?;
    a.need("--model")?;
    let app = open_session(&a, out)?;
    let decoder = load_decoder(&a, &app, out)?;
    let server = start(decoder, &cfg)?;
    let addr = server.addr;
    say(out, format!("ferrum-gpu server: http://{addr}/"));
    if cfg.token.is_some() {
        say(
            out,
            "auth: bearer token required (GET /health is open)".to_string(),
        );
    }
    Ok(Booted {
        server,
        addr,
        has_token: cfg.token.is_some(),
    })
}
