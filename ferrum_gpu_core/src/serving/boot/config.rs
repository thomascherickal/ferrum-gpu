use super::*;

/// The flags of `ferrum-gpu-server`.
pub(super) fn parse_flags(args: &[String]) -> Result<Args, String> {
    let flags = [
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
    ];
    Args::parse(&flags, args)
}

/// The listen address, token and limits; refuses an exposed server with no token.
pub(super) fn server_config(a: &Args) -> Result<Config, String> {
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
        ..Config::default()
    };
    check_exposure(
        &cfg.host,
        cfg.token.as_deref(),
        a.has("--allow-unauthenticated"),
    )?;
    Ok(cfg)
}
