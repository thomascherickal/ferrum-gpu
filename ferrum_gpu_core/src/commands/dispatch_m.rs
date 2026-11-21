use super::*;

/// The command switch without the test-only lock, so one command (`run`) can invoke others: the
/// lock is not reentrant, and taking it twice on one thread deadlocks.
pub(super) fn dispatch(args: &[String], out: &mut dyn Write) -> Result<(), String> {
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    let w = |e: std::io::Error| e.to_string();
    match cmd.as_str() {
        "-h" | "--help" | "help" => writeln!(out, "{USAGE}").map_err(w),
        "info" => info(
            &Args::parse(&table(&[&DEVICE, &[val("--model")]]), rest)?,
            out,
        ),
        "train" => {
            let a = Args::parse(
                &table(&[
                    &DEVICE,
                    &TRAIN,
                    &gpt2::SIZE_FLAGS,
                    &gpt2::TRAIN_FLAGS,
                    &[val("--quant"), val("--arch"), val("--epochs")],
                ]),
                rest,
            )?;
            match a.get("--arch").unwrap_or("gpt2") {
                "gpt2" => gpt2::train(&a, out, None),
                "llama" => {
                    if let Some(f) = gpt2::only_gpt2(&a) {
                        return Err(format!("{f} needs --arch gpt2"));
                    }
                    train(&a, out, false)
                }
                o => Err(format!("--arch: unknown '{o}' (gpt2, llama)")),
            }
        }
        "lab" => lab::run(&Args::parse(&lab::FLAGS, rest)?, out),
        "run" => combined::run_combined(rest, out),
        "finetune-gguf" => {
            let a = Args::parse(
                &table(&[
                    &DEVICE,
                    &TRAIN,
                    &gpt2::TRAIN_FLAGS,
                    &[val("--model"), val("--quant"), val("--epochs")],
                ]),
                rest,
            )?;
            let model = a.need("--model")?;
            if gpt2::is_gpt2(model) {
                gpt2::train(&a, out, Some(gpt2::load(model)?))
            } else {
                if let Some(f) = gpt2::only_gpt2(&a) {
                    return Err(format!("{f} needs a gpt2 model"));
                }
                train(&a, out, true)
            }
        }
        "generate" | "run-gguf" => {
            let a = Args::parse(
                &table(&[
                    &DEVICE,
                    &GEN,
                    &gpt2::GEN_FLAGS,
                    &[val("--seed"), val("--max-new"), switch("--raw")],
                ]),
                rest,
            )?;
            if gpt2::is_gpt2(a.need("--model")?) {
                gpt2::generate(&a, out, cmd == "run-gguf")
            } else if llm::has_tokenizer(a.need("--model")?) {
                llm::generate(&a, out, cmd == "run-gguf")
            } else {
                generate(&a, out, cmd == "run-gguf")
            }
        }
        "prepare" => gpt2::prepare::run(
            &Args::parse(&gpt2::prepare::PREPARE_FLAGS, rest)?,
            rest,
            out,
        ),
        "tokenize" => gpt2::tokenize(
            &Args::parse(
                &[
                    val("--data"),
                    val("--tokenizer"),
                    val("--out"),
                    val("--doc-sep"),
                ],
                rest,
            )?,
            out,
        ),
        "export-ollama" => runtimes::export_ollama(
            &Args::parse(
                &[
                    val("--model"),
                    val("--name"),
                    val("--bin"),
                    val("--modelfile-only"),
                ],
                rest,
            )?,
            out,
        ),
        "export-lmstudio" => runtimes::export_lmstudio(
            &Args::parse(&[val("--model"), val("--repo"), val("--bin")], rest)?,
            out,
        ),
        "export-gguf" => export_gguf(
            &Args::parse(&[val("--model"), val("--out"), val("--quant")], rest)?,
            out,
        ),
        // eval-gguf is ferrum-cpu's name for scoring an imported model; ours scores any file
        "eval" | "eval-gguf" => {
            let a = Args::parse(
                &table(&[&[
                    val("--model"),
                    val("--data"),
                    val("--ctx"),
                    val("--json"),
                    val("--quant"),
                ]]),
                rest,
            )?;
            if gpt2::is_gpt2(a.need("--model")?) {
                gpt2::eval(&a, out)
            } else if llm::has_tokenizer(a.need("--model")?) {
                llm::eval(&a, out)
            } else {
                eval(&a, out)
            }
        }
        other => Err(format!("unknown command '{other}'\n{USAGE}")),
    }
}

/// Build the app (device) from the shared device flags.
pub(super) fn app(a: &Args, work: Workload) -> Result<Session, String> {
    app_requested(
        a,
        work,
        DeviceChoice::parse(a.get("--device").unwrap_or("auto"))?,
    )
}

/// [`app`] with the device choice decided by the caller instead of read from `--device`.
pub(super) fn app_requested(
    a: &Args,
    work: Workload,
    requested: DeviceChoice,
) -> Result<Session, String> {
    let threads = a.num(
        "--threads",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )?;
    let d = decide(requested, work, &Gpu::list(), a.get("--gpu-adapter"))?;
    let weights = gpu_weights(a.get("--gpu-weights"))?;
    let mut app = Session::new_with(d.choice, d.adapter.as_deref(), threads.max(1), weights)?;
    if d.note.is_some() {
        app.fallback_note = d.note;
    }
    Ok(app)
}
