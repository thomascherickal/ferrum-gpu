//! The jobs API against the real `ferrum-gpu` binary: the progress lines the page parses are the
//! ones the CLI prints today, and a polite stop (the `.stop` file) ends a real training cleanly.
//! Skips, with the reason printed, when the binary has not been built (`cargo build -p ferrum_gpu_cli`).

use ferrum_gpu_gui::jobs::progress::Event;
use ferrum_gpu_gui::jobs::{Job, Jobs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `target/<profile>/ferrum-gpu`, next to the `deps` directory this test runs from.
fn cli() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let p = exe.parent()?.parent()?.join(if cfg!(windows) {
        "ferrum-gpu.exe"
    } else {
        "ferrum-gpu"
    });
    p.is_file().then_some(p)
}

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

#[test]
fn a_real_training_run_streams_steps_and_validation_and_writes_the_model() {
    let Some(exe) = cli() else {
        eprintln!("SKIP: target/<profile>/ferrum-gpu is not built");
        return;
    };
    let dir = work_dir("full");
    let jobs = Jobs::new(Some(exe));
    let job = jobs
        .start("train", args(&dir, &["--steps", "30", "--val", "0.2"]))
        .unwrap();
    wait(&job, 120);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("done", Some(0)), "{:?}", p.lines);
    let steps: Vec<_> = p
        .lines
        .iter()
        .filter_map(|(_, l)| match &l.ev {
            Some(Event::Step {
                step, total, loss, ..
            }) => Some((*step, *total, *loss)),
            _ => None,
        })
        .collect();
    assert_eq!(steps.iter().map(|s| s.0).collect::<Vec<_>>(), [10, 20, 30]);
    assert!(steps.iter().all(|s| s.1 == 30));
    // each reported loss is one batch's, so it is noisy on a run this short: check it is a real number
    assert!(
        steps.iter().all(|s| s.2.is_finite() && s.2 > 0.0),
        "{steps:?}"
    );
    assert!(
        p.lines
            .iter()
            .any(|(_, l)| matches!(l.ev, Some(Event::Val { .. }))),
        "validation lines were parsed"
    );
    assert!(p.lines.iter().any(|(_, l)| l.text.starts_with("wrote ")));
    assert!(dir.join("m.gguf").is_file());
}

#[test]
fn a_real_training_run_stops_cleanly_when_asked() {
    let Some(exe) = cli() else {
        eprintln!("SKIP: target/<profile>/ferrum-gpu is not built");
        return;
    };
    let dir = work_dir("stop");
    let jobs = Jobs::new(Some(exe));
    let job = jobs
        .start("train", args(&dir, &["--steps", "200000"]))
        .unwrap();
    let t0 = Instant::now();
    while job.poll(0).last_step.is_none() {
        assert!(
            t0.elapsed() < Duration::from_secs(120),
            "{:?}",
            job.poll(0).lines
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(job.stop(false), "a polite stop was accepted");
    wait(&job, 60);
    let p = job.poll(0);
    assert_eq!(p.state, "stopped", "{:?}", p.lines);
    assert_eq!(p.exit, Some(0), "the CLI ended itself: {:?}", p.lines);
    assert!(
        p.lines.iter().any(|(_, l)| l.text.contains(".stop found")),
        "{:?}",
        p.lines
    );
    assert!(dir.join("m.gguf").is_file(), "the partial model was kept");
    assert!(!dir.join("m.stop").exists(), "the stop file was cleaned up");
}
