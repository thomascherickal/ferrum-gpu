use super::*;

/// Start the server for `args` (the flags of `ferrum-gpu-server`), writing its status lines to `out`.
pub fn boot(args: &[String], out: &mut dyn Write) -> Result<Booted, String> {
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
        ..Config::default()
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
    let mut app = Session::new(dec.choice, dec.adapter.as_deref(), threads.max(1))?;
    if dec.note.is_some() {
        app.fallback_note = dec.note;
    }
    say(
        out,
        format!("device: {} ({})", app.backend.name(), app.backend.kind()),
    );
    if let Some(n) = &app.fallback_note {
        say(out, format!("note: {n}"));
    }
    let mem = a
        .get("--gpu-mem-mb")
        .map(|_| a.num("--gpu-mem-mb", 0u64))
        .transpose()?;
    let decoder: Arc<dyn Decoder> = if gpt2::is_gpt2(path) {
        let l = gpt2::load(path)?;
        crate::commands::gate_with(
            &crate::commands::sysmem::System,
            out,
            &app,
            &crate::commands::preflight::gpt2_decode_buffers(&l.model),
            mem,
        )?;
        Arc::new(Gpt2Decoder::new(app.backend.clone(), l))
    } else {
        let model = crate::commands::load(path)?;
        crate::commands::gate_with(
            &crate::commands::sysmem::System,
            out,
            &app,
            &crate::commands::preflight::decode_buffers_for(&model, &app),
            mem,
        )?;
        Arc::new(ModelDecoder::new(app.backend.clone(), model))
    };
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
