use super::*;

#[test]
fn the_js_reader_repack_and_rope_tables_match_the_rust_side() {
    if Command::new("node").arg("--version").output().is_err() {
        eprintln!("SKIP: node is not installed");
        return;
    }
    let server = start(app(), 0).unwrap();
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/js_tests/reader.test.mjs");
    // a hung node is ended by the deadline instead of hanging the suite
    let out = common::with_deadline("node", 120)
        .arg(script)
        .arg(format!("http://127.0.0.1:{}", server.addr.port()))
        .output()
        .unwrap();
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    eprintln!("{stdout}");
    assert!(
        out.status.success(),
        "node tests failed:\n{stdout}\n{stderr}"
    );
    assert!(stdout.contains("OK "), "{stdout}");
    server.stop();
}

#[test]
fn the_in_browser_runtime_matches_the_cpu_oracle_in_headless_chrome() {
    let Some(chrome) = common::chrome() else {
        eprintln!("SKIP: no Chrome or Chromium installed");
        return;
    };
    let server = start(app(), 0).unwrap();
    let port = server.addr.port();
    let browser = common::Browser::open(
        &chrome,
        &format!("http://127.0.0.1:{port}/?autotest=webgpu"),
        "gpu",
        300,
    );
    let Some(text) = common::wait_for(Duration::from_secs(120), || common::selftest_result(port))
    else {
        panic!(
            "the page never reported a result; browser console:\n{}",
            browser.console()
        );
    };
    eprintln!("browser reported: {text}");
    let j = Json::parse(&text).unwrap();
    if j.get("skipped") == Some(&Json::Bool(true)) {
        eprintln!(
            "SKIP: WebGPU is not usable in this Chrome: {}",
            j.get("reason").and_then(Json::as_str).unwrap_or("?")
        );
        return;
    }
    assert_eq!(j.get("pass"), Some(&Json::Bool(true)), "{text}");
    let Some(Json::Arr(adapters)) = j.get("adapters") else {
        panic!("no adapters in {text}")
    };
    assert!(!adapters.is_empty());
    for a in adapters {
        let who = a.get("adapter").map(|d| d.to_string()).unwrap_or_default();
        assert_eq!(a.get("pass"), Some(&Json::Bool(true)), "{who}: {a}");
        let Some(Json::Arr(cases)) = a.get("cases") else {
            panic!("no cases for {who}")
        };
        assert_eq!(cases.len(), 6, "{who}");
        for c in cases {
            assert_eq!(c.get("ok"), Some(&Json::Bool(true)), "{who}: {c}");
        }
    }
    server.stop();
}
