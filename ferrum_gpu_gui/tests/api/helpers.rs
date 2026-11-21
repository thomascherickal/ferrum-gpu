use super::http::get;
use ferrum_gpu_gui::app::{App, DeviceChoice};
use ferrum_gpu_gui::engine::CpuBackend;
use ferrum_gpu_gui::json::Json;
use ferrum_gpu_gui::server::{start, Running};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(super) fn serve(app: App) -> (Running, Arc<App>) {
    let app = Arc::new(app);
    (start(app.clone(), 0).unwrap(), app)
}

pub(super) fn cpu_app() -> App {
    App::with_backend(
        Arc::new(CpuBackend::new(2)),
        DeviceChoice::Cpu,
        None,
        vec![],
    )
}

pub(super) fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

pub(super) fn wait_until_done(port: u16) -> Json {
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

pub(super) fn losses(s: &Json) -> Vec<f64> {
    match s.get("losses") {
        Some(Json::Arr(a)) => a.iter().filter_map(Json::as_f64).collect(),
        _ => vec![],
    }
}
