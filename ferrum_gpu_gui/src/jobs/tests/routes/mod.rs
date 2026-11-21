use crate::app::{App, DeviceChoice};
use crate::engine::CpuBackend;
use crate::http::Request;
use crate::jobs::routes::{handles, route};
use crate::jobs::{Jobs, RunFn};
use crate::json::Json;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Fake commands: `info` echoes its arguments, `train` reports progress then waits to be stopped,
/// `eval` fails.
fn fake() -> Jobs {
    let run: Arc<RunFn> = Arc::new(|argv, out| {
        match argv[0].as_str() {
            "info" => {
                writeln!(out, "args: {}", argv[1..].join(" ")).unwrap();
                writeln!(out, "device: fake").unwrap();
            }
            "train" => {
                writeln!(
                    out,
                    "step 10/100 loss 3.0000 grad_norm 0.500 tokens/s 1000 elapsed 1.0s eta 9s"
                )
                .unwrap();
                writeln!(out, "epoch 1 val loss 2.5000 (best)").unwrap();
                writeln!(
                    out,
                    "step 20/100 loss 2.0000 grad_norm 0.400 tokens/s 1100 elapsed 2.0s eta 8s"
                )
                .unwrap();
                let t0 = Instant::now();
                while !ferrum_gpu_core::cancel::requested() {
                    assert!(t0.elapsed() < Duration::from_secs(20), "never stopped");
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
            _ => return Err("the fake command failed".into()),
        }
        Ok(())
    });
    Jobs::with_runner(run)
}

fn app() -> Arc<App> {
    Arc::new(
        App::with_backend(
            Arc::new(CpuBackend::new(1)),
            DeviceChoice::Cpu,
            None,
            vec![],
        )
        .with_jobs(fake()),
    )
}

fn call(a: &App, method: &str, path: &str, query: &[(&str, &str)], body: &[u8]) -> (u16, Json) {
    let req = Request {
        method: method.into(),
        path: path.into(),
        query: query
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        headers: vec![],
        body: body.to_vec(),
    };
    let r = route(a, &req);
    (
        r.status,
        Json::parse(&String::from_utf8_lossy(&r.body)).unwrap_or(Json::Null),
    )
}

fn field(j: &Json, k: &str) -> Json {
    j.get(k).cloned().unwrap_or(Json::Null)
}

fn wait_state(a: &App, id: u64, want: &str) -> Json {
    let t0 = Instant::now();
    loop {
        let (s, j) = call(a, "GET", &format!("/api/jobs/{id}"), &[], b"");
        assert_eq!(s, 200);
        if field(&j, "state").as_str() == Some(want) {
            return j;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "never {want}: {j}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

mod a_failing_command_is_an_error_job_with_the_message_as_a_line;
mod only_the_jobs_paths_are_handled;
mod the_api_router_sends_jobs_paths_here;
