use super::*;

#[test]
fn the_new_tabs_work_in_a_real_browser_with_the_real_commands() {
    let Some(chrome) = common::chrome() else {
        eprintln!("SKIP: no Chrome or Chromium installed");
        return;
    };
    let dir = work("tabs");
    let inputs = corpus(&dir);
    // a tokenizer file, as a training run saves it
    let tok = dir.join("tok.gguf");
    let train: Vec<String> = [
        "train", "--preset", "tiny", "--steps", "2", "--batch", "2", "--vocab", "300", "--device",
        "cpu", "--data",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([
        dir.join("corpus.txt").display().to_string(),
        "--out".into(),
        dir.join("m.gguf").display().to_string(),
        "--tokenizer".into(),
        tok.display().to_string(),
    ])
    .collect();
    ferrum_gpu_core::commands::run(&train, &mut Vec::new()).unwrap();
    assert!(tok.is_file());

    let app = Arc::new(App::with_backend(
        Arc::new(CpuBackend::new(2)),
        DeviceChoice::Cpu,
        None,
        vec![],
    ));
    let server = start(app, 0).unwrap();
    let port = server.addr.port();
    let out = dir.join("prepared");
    let tokout = dir.join("tokens.gguf");
    let gtout = dir.join("trained.gguf");
    let gtstop = dir.join("stopped.gguf");
    let ftout = dir.join("tuned.gguf");
    let exgout = dir.join("tuned-int4.gguf");
    let exomf = dir.join("Modelfile");
    let q = |k: &str, v: &Path| format!("{k}={}", v.display());
    let url = format!(
        "http://127.0.0.1:{port}/?autotest=studio&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}",
        q("file", &dir.join("corpus.txt")),
        q("dir", &dir),
        format_args!("inputs={inputs}"),
        q("out", &out),
        q("tok", &tok),
        q("tokout", &tokout),
        q("gtout", &gtout),
        q("gtstop", &gtstop),
        q("ftout", &ftout),
        q("exgout", &exgout),
        q("exomf", &exomf),
    );
    let browser = common::Browser::open(&chrome, &url, "studio", 240);
    let Some(result) = common::wait_for(Duration::from_secs(150), || fetch_result(port)) else {
        panic!(
            "the page never reported a result; browser console:\n{}",
            browser.console()
        );
    };
    eprintln!("page reported: {result}");
    let j = Json::parse(&result).unwrap();
    assert_eq!(j.get("kind").and_then(Json::as_str), Some("studio"));
    assert_eq!(j.get("pass"), Some(&Json::Bool(true)), "{result}");
    for f in ["train.txt", "val.txt", "test.txt", "manifest.json"] {
        assert!(out.join(f).is_file(), "prepare wrote {f}");
    }
    assert!(
        tokout.metadata().unwrap().len() > 0,
        "tokenize wrote the token file"
    );
    assert!(
        gtout.metadata().unwrap().len() > 0,
        "the Train tab saved a model"
    );
    assert!(
        gtstop.metadata().unwrap().len() > 0,
        "a stopped run still saved the model it had"
    );
    assert!(
        ftout.metadata().unwrap().len() > 0,
        "the Fine-tune tab saved a model"
    );
    assert!(
        exgout.metadata().unwrap().len() < ftout.metadata().unwrap().len(),
        "the int4 export is smaller than the int8 model"
    );
    assert!(
        std::fs::read_to_string(&exomf).unwrap().contains("FROM "),
        "the Modelfile names the model"
    );
    server.stop();
}
