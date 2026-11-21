//! The GUI's API end to end over real TCP: the same flow on the CPU and the GPU backend.
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::{CpuBackend, GpuBackend};
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::{start, Running};
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

mod a_run_can_be_stopped_and_keeps_its_partial_weights;
mod bad_input_gets_a_clear_error_and_never_a_crash;
mod info_describes_the_device_the_models_and_the_limits;
mod models_move_through_gguf_in_every_precision;
mod the_journey_works_on_the_cpu_backend;
