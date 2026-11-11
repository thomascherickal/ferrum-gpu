//! The GUI's API end to end over real TCP: the same flow on the CPU and the GPU backend.

use gpu_gui::app::{App, DeviceChoice};
use gpu_gui::engine::{CpuBackend, GpuBackend};
use gpu_gui::json::Json;
use gpu_gui::server::{start, Running};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Reply {
    status: u16,
    head: String,
    body: Vec<u8>,
}

impl Reply {
    fn json(&self) -> Json {
        Json::parse(std::str::from_utf8(&self.body).expect("utf-8"))
            .unwrap_or_else(|e| panic!("{e}: {:?}", String::from_utf8_lossy(&self.body)))
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

fn raw(port: u16, request: &[u8]) -> Reply {
    let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    c.write_all(request).unwrap();
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    let split = out
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("a response head");
    let head = String::from_utf8_lossy(&out[..split]).into_owned();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    Reply {
        status,
        head,
        body: out[split + 4..].to_vec(),
    }
}

fn call(port: u16, method: &str, path: &str, body: &[u8]) -> Reply {
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    req.extend_from_slice(body);
    raw(port, &req)
}

fn post(port: u16, path: &str, body: &str) -> Reply {
    call(port, "POST", path, body.as_bytes())
}

fn get(port: u16, path: &str) -> Reply {
    call(port, "GET", path, b"")
}

fn serve(app: App) -> (Running, Arc<App>) {
    let app = Arc::new(app);
    (start(app.clone(), 0).unwrap(), app)
}

fn cpu_app() -> App {
    App::with_backend(
        Arc::new(CpuBackend::new(2)),
        DeviceChoice::Cpu,
        None,
        vec![],
    )
}

fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

fn wait_until_done(port: u16) -> Json {
    let t0 = Instant::now();
    loop {
        let s = get(port, "/api/train/status").json();
        if field(&s, "state").as_str() != Some("running") {
            return s;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(120),
            "training did not finish"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn losses(s: &Json) -> Vec<f64> {
    match s.get("losses") {
        Some(Json::Arr(a)) => a.iter().filter_map(Json::as_f64).collect(),
        _ => vec![],
    }
}

/// The whole user journey, identical for every backend.
fn journey(port: u16) {
    assert_eq!(
        get(port, "/api/info").json().get("model"),
        Some(&Json::Null),
        "no model at first"
    );
    assert_eq!(
        post(port, "/api/generate", r#"{"prompt":"hi"}"#).status,
        400,
        "generate needs a model"
    );
    assert_eq!(
        post(port, "/api/train", r#"{"text":"abc"}"#).status,
        400,
        "train needs a model"
    );
    let created = post(port, "/api/model", r#"{"preset":"tiny","seed":7}"#);
    assert_eq!(created.status, 200, "{}", created.text());
    assert_eq!(field(&created.json(), "preset").as_str(), Some("tiny"));
    let text = "the quick brown fox jumps over the lazy dog. ".repeat(14);
    let start = post(
        port,
        "/api/train",
        &Json::obj([
            ("text", Json::Str(text)),
            ("steps", Json::Num(60.0)),
            ("lr", Json::Num(0.01)),
            ("batch", Json::Num(4.0)),
            ("context", Json::Num(24.0)),
        ])
        .to_string(),
    );
    assert_eq!(start.status, 200, "{}", start.text());
    let busy = post(port, "/api/train", r#"{"text":"another run"}"#);
    let status_now = get(port, "/api/train/status").json();
    if field(&status_now, "state").as_str() == Some("running") {
        assert_eq!(
            busy.status, 409,
            "a second run is refused while one is going"
        );
        assert_eq!(
            post(port, "/api/model", "{}").status,
            409,
            "so is replacing the model"
        );
    }
    let done = wait_until_done(port);
    assert_eq!(field(&done, "state").as_str(), Some("done"), "{done}");
    let l = losses(&done);
    assert_eq!(l.len(), 60);
    assert!(l[59] < 0.5 * l[0], "loss {} -> {}", l[0], l[59]);
    let g = post(
        port,
        "/api/generate",
        r#"{"prompt":"the quick brown ","max_new":12}"#,
    );
    assert_eq!(g.status, 200, "{}", g.text());
    let out = g.json();
    let words = field(&out, "text").as_str().unwrap().to_string();
    assert!(words.starts_with("fox"), "got {words:?}");
    assert_eq!(
        field(&out, "tokens").to_string().matches(',').count(),
        11,
        "12 tokens"
    );
}

#[test]
fn the_journey_works_on_the_cpu_backend() {
    let (srv, _app) = serve(cpu_app());
    journey(srv.addr.port());
    srv.stop();
}

#[test]
fn the_same_journey_works_on_the_gpu_backend() {
    let Some(g) = ferrum_wgpu::device::open_for_test() else {
        return;
    };
    let app = App::with_backend(
        Arc::new(GpuBackend::new(Arc::new(g))),
        DeviceChoice::Gpu,
        None,
        ferrum_wgpu::Gpu::list(),
    );
    let (srv, _app) = serve(app);
    journey(srv.addr.port());
    let info = get(srv.addr.port(), "/api/info").json();
    assert_eq!(
        field(&field(&info, "device"), "active").as_str(),
        Some("gpu")
    );
    srv.stop();
}

#[test]
fn static_files_are_served_from_the_binary_and_nothing_else() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let page = get(p, "/");
    assert_eq!(page.status, 200);
    assert!(page.head.contains("text/html") && page.text().contains("<title>ferrum-gpu</title>"));
    assert!(get(p, "/index.html").text().contains("role=\"tablist\""));
    assert!(get(p, "/app.js").head.contains("text/javascript"));
    assert!(get(p, "/style.css").head.contains("text/css"));
    assert!(get(p, "/shaders/prelude.wgsl").text().contains("fn gid"));
    assert!(get(p, "/shaders/llama/rmsnorm.wgsl")
        .text()
        .contains("fn main"));
    for missing in [
        "/nope",
        "/shaders/llama/nope.wgsl",
        "/shaders/../Cargo.toml",
        "/shaders/llama/../fwd/add.wgsl",
        "/web/nothing.js",
        "/shaders/",
        "/etc/passwd",
        "/%2e%2e/Cargo.toml",
    ] {
        assert_eq!(get(p, missing).status, 404, "{missing}");
    }
    assert_eq!(call(p, "DELETE", "/api/info", b"").status, 405);
    assert_eq!(call(p, "POST", "/api/nothing", b"{}").status, 405);
    assert_eq!(call(p, "POST", "/", b"").status, 405);
    srv.stop();
}

#[test]
fn the_loopback_guard_and_request_limits_apply_through_the_real_server() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    assert_eq!(
        raw(p, b"GET /api/info HTTP/1.1\r\nHost: evil.example\r\n\r\n").status,
        403
    );
    let cross = format!(
        "GET /api/info HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nOrigin: http://evil.example\r\n\r\n"
    );
    assert_eq!(raw(p, cross.as_bytes()).status, 403);
    assert_eq!(raw(p, b"junk\r\n\r\n").status, 400);
    let huge = format!("PUT /api/model.gguf HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nContent-Length: 999999999999\r\n\r\n");
    assert_eq!(raw(p, huge.as_bytes()).status, 413);
    srv.stop();
}

#[test]
fn bad_input_gets_a_clear_error_and_never_a_crash() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let err = |r: Reply| -> String { field(&r.json(), "error").as_str().unwrap_or("").to_string() };
    assert!(err(post(p, "/api/model", "{")).contains("bad JSON"));
    assert!(err(post(p, "/api/model", r#"{"preset":"huge"}"#)).contains("unknown preset"));
    assert!(err(post(p, "/api/model", r#"{"seed":-1}"#)).contains("seed"));
    assert!(err(call(p, "POST", "/api/model", &[0xff, 0xfe])).contains("UTF-8"));
    assert_eq!(post(p, "/api/model", r#"{"preset":"tiny"}"#).status, 200);
    assert!(err(post(p, "/api/train", "{}")).contains("'text'"));
    assert!(err(post(p, "/api/train", r#"{"text":"short"}"#)).contains("needs more"));
    assert!(err(post(
        p,
        "/api/train",
        &format!(r#"{{"text":"{}","steps":0}}"#, "x".repeat(100))
    ))
    .contains("steps"));
    assert!(err(post(
        p,
        "/api/train",
        &format!(r#"{{"text":"{}","lr":"fast"}}"#, "x".repeat(100))
    ))
    .contains("lr"));
    assert_eq!(
        post(
            p,
            "/api/train",
            &format!(r#"{{"text":"{}"}}"#, "x".repeat((1 << 20) + 1))
        )
        .status,
        413
    );
    assert!(err(post(p, "/api/generate", r#"{"prompt":""}"#)).contains("empty"));
    assert!(err(post(p, "/api/generate", "{}")).contains("'prompt'"));
    assert_eq!(
        post(
            p,
            "/api/generate",
            &format!(r#"{{"prompt":"{}"}}"#, "y".repeat(5000))
        )
        .status,
        413
    );
    assert!(err(post(p, "/api/generate", r#"{"prompt":"a","max_new":0}"#)).contains("max_new"));
    assert!(err(post(
        p,
        "/api/generate",
        r#"{"prompt":"a","temperature":99}"#
    ))
    .contains("temperature"));
    assert!(err(post(
        p,
        "/api/generate",
        &format!(r#"{{"prompt":"{}"}}"#, "z".repeat(200))
    ))
    .contains("context"));
    assert!(err(get(p, "/api/model.gguf?quant=q3")).contains("unknown quant"));
    assert_eq!(
        post(p, "/api/train/stop", "{}").json().get("stopping"),
        Some(&Json::Bool(false))
    );
    assert_eq!(
        field(&get(p, "/api/train/status").json(), "state").as_str(),
        Some("idle")
    );
    srv.stop();
}

#[test]
fn a_run_can_be_stopped_and_keeps_its_partial_weights() {
    let (srv, app) = serve(cpu_app());
    let p = srv.addr.port();
    post(p, "/api/model", r#"{"preset":"tiny"}"#);
    let before = gpu_gui::app::lock(&app.model).clone().unwrap();
    let body = format!(
        r#"{{"text":"{}","steps":100000,"batch":8,"context":32}}"#,
        "ab".repeat(200)
    );
    assert_eq!(post(p, "/api/train", &body).status, 200);
    let t0 = Instant::now();
    while field(&get(p, "/api/train/status").json(), "step")
        .as_usize()
        .unwrap_or(0)
        < 3
    {
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        post(p, "/api/train/stop", "{}").json().get("stopping"),
        Some(&Json::Bool(true))
    );
    let s = wait_until_done(p);
    assert_eq!(field(&s, "state").as_str(), Some("stopped"));
    assert!(field(&s, "step").as_usize().unwrap() < 100_000);
    assert_ne!(
        gpu_gui::app::lock(&app.model).clone().unwrap(),
        before,
        "the partial weights were kept"
    );
    srv.stop();
}

#[test]
fn a_failed_run_is_reported_and_the_model_is_untouched() {
    // a model whose head is tied cannot be trained: the run fails, with the reason
    let (srv, app) = serve(cpu_app());
    let p = srv.addr.port();
    let mut tied = gpu_gui::engine::fresh_model("tiny", 1).unwrap();
    tied = ferrum_ref::llama::Llama { tied: true, ..tied };
    *gpu_gui::app::lock(&app.model) = Some(tied.clone());
    let body = format!(r#"{{"text":"{}","steps":3}}"#, "ab".repeat(100));
    assert_eq!(post(p, "/api/train", &body).status, 200);
    let s = wait_until_done(p);
    assert_eq!(field(&s, "state").as_str(), Some("error"));
    assert!(field(&s, "error").as_str().unwrap().contains("tied"), "{s}");
    assert_eq!(gpu_gui::app::lock(&app.model).clone().unwrap(), tied);
    srv.stop();
}

#[test]
fn models_move_through_gguf_in_every_precision() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    assert_eq!(
        get(p, "/api/model.gguf").status,
        404,
        "nothing to export yet"
    );
    post(p, "/api/model", r#"{"preset":"small","seed":3}"#);
    let gen = |port: u16| post(port, "/api/generate", r#"{"prompt":"hello ","max_new":10}"#).json();
    let want = gen(p);
    let f32_bytes = get(p, "/api/model.gguf").body;
    assert!(f32_bytes.starts_with(b"GGUF"));
    let (q8, q4) = (
        get(p, "/api/model.gguf?quant=q8_0"),
        get(p, "/api/model.gguf?quant=q4_0"),
    );
    assert!(
        q8.head.contains("attachment")
            && q8.body.len() < f32_bytes.len() / 2
            && q4.body.len() < q8.body.len()
    );
    // the exact export loads into a different server and decodes identically
    let (srv2, _a2) = serve(cpu_app());
    let p2 = srv2.addr.port();
    let up = call(p2, "PUT", "/api/model.gguf", &f32_bytes);
    assert_eq!(up.status, 200, "{}", up.text());
    assert_eq!(field(&up.json(), "preset").as_str(), Some("small"));
    assert_eq!(gen(p2), want, "an f32 round trip is exact");
    // quantised exports load too (a different model, so only shape and sanity are checked)
    for q in [&q8, &q4] {
        let up = call(p2, "PUT", "/api/model.gguf", &q.body);
        assert_eq!(up.status, 200, "{}", up.text());
        assert_eq!(
            field(&gen(p2), "tokens").to_string().matches(',').count(),
            9
        );
    }
    // garbage and truncated files are refused with a reason
    for bad in [&b"not a gguf"[..], &f32_bytes[..f32_bytes.len() / 2], b""] {
        let r = call(p2, "PUT", "/api/model.gguf", bad);
        assert_eq!(r.status, 400, "{}", r.text());
        assert!(field(&r.json(), "error")
            .as_str()
            .unwrap()
            .contains("usable"));
    }
    srv.stop();
    srv2.stop();
}

#[test]
fn a_model_with_another_vocabulary_loads_but_refuses_text() {
    use ferrum_model::gguf::gpt2::Quant;
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    let mut cfg = gpu_gui::engine::preset("tiny").unwrap();
    cfg.vocab = 300;
    let m = ferrum_ref::llama::Llama::random(cfg, 2, false, false);
    let bytes = ferrum_model::gguf::llama::to_gguf(&m, Quant::F32);
    let up = call(p, "PUT", "/api/model.gguf", &bytes);
    assert_eq!(up.status, 200);
    assert_eq!(field(&up.json(), "byte_tokens"), Json::Bool(false));
    let g = post(p, "/api/generate", r#"{"prompt":"hi"}"#);
    assert_eq!(g.status, 400);
    assert!(g.text().contains("byte-level"));
    srv.stop();
}

#[test]
fn info_describes_the_device_the_models_and_the_limits() {
    let note = Some("no usable GPU (test); using the CPU".to_string());
    let app = App::with_backend(
        Arc::new(CpuBackend::new(1)),
        DeviceChoice::Auto,
        note,
        vec![],
    );
    let (srv, _app) = serve(app);
    let p = srv.addr.port();
    let info = get(p, "/api/info").json();
    let dev = field(&info, "device");
    assert_eq!(field(&dev, "requested").as_str(), Some("auto"));
    assert_eq!(field(&dev, "active").as_str(), Some("cpu"));
    assert!(field(&dev, "note")
        .as_str()
        .unwrap()
        .contains("using the CPU"));
    assert!(matches!(field(&info, "adapters"), Json::Arr(a) if a.is_empty()));
    assert!(matches!(field(&info, "presets"), Json::Arr(a) if a.len() == 3));
    assert!(
        field(&field(&info, "limits"), "max_new")
            .as_usize()
            .unwrap()
            >= 64
    );
    post(p, "/api/model", r#"{"preset":"medium"}"#);
    let m = field(&get(p, "/api/info").json(), "model");
    assert_eq!(field(&m, "preset").as_str(), Some("medium"));
    assert_eq!(field(&m, "byte_tokens"), Json::Bool(true));
    assert!(field(&m, "params").as_usize().unwrap() > 100_000);
    srv.stop();
}

#[test]
fn the_app_picks_a_backend_for_auto_cpu_and_a_missing_adapter() {
    let cpu = App::new(DeviceChoice::Cpu, None, 1).unwrap();
    assert_eq!(
        (cpu.backend.kind(), cpu.fallback_note.is_none()),
        ("cpu", true)
    );
    let auto = App::new(DeviceChoice::Auto, Some("no-such-adapter"), 1).unwrap();
    assert_eq!(auto.backend.kind(), "cpu");
    assert!(auto
        .fallback_note
        .as_deref()
        .unwrap()
        .contains("using the CPU"));
    assert!(App::new(DeviceChoice::Gpu, Some("no-such-adapter"), 1).is_err());
}

#[test]
fn the_browser_self_test_result_is_stored_and_returned() {
    let (srv, _app) = serve(cpu_app());
    let p = srv.addr.port();
    assert_eq!(get(p, "/api/selftest/result").status, 404);
    assert_eq!(
        post(p, "/api/selftest/result", r#"{"pass":true}"#).status,
        200
    );
    assert_eq!(get(p, "/api/selftest/result").text(), r#"{"pass":true}"#);
    srv.stop();
}
