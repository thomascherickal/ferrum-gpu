//! The jobs API against the real commands, in this process: the progress lines the page parses are
//! the ones the commands print today, and a stop ends a real training cleanly with the model it has
//! so far. No binary is involved, so these tests never skip.
use ferrum_gpu_gui::jobs::progress::Event;
use ferrum_gpu_gui::jobs::{Job, Jobs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn work_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_jobs_real_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn args(dir: &Path, extra: &[&str]) -> Vec<String> {
    let corpus = dir.join("c.txt");
    std::fs::write(
        &corpus,
        "the quick brown fox jumps over the lazy dog. ".repeat(90),
    )
    .unwrap();
    let mut a: Vec<String> = [
        "--preset",
        "tiny",
        "--data",
        corpus.to_str().unwrap(),
        "--out",
        dir.join("m.gguf").to_str().unwrap(),
        "--batch",
        "4",
        "--vocab",
        "300",
        "--device",
        "cpu",
        "--verbose",
    ]
    .map(String::from)
    .to_vec();
    a.extend(extra.iter().map(|s| s.to_string()));
    a
}

fn wait(job: &Arc<Job>, secs: u64) {
    let t0 = Instant::now();
    while !job.is_finished() {
        assert!(
            t0.elapsed() < Duration::from_secs(secs),
            "{:?}",
            job.poll(0).lines
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

mod a_real_training_run_streams_steps_and_validation_and_writes_the_model;
