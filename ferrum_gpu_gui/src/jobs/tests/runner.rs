use super::support::{args, exists, jobs, stub, texts, wait, wait_for_line};
use crate::jobs::progress::Event;
use crate::jobs::{Jobs, MAX_KEPT, MAX_LINES, MAX_RUNNING};
use std::time::Duration;

#[test]
fn output_is_streamed_line_by_line_and_a_clean_exit_is_done() {
    let (jobs, _d) = jobs();
    let job = jobs.start("info", args(&["--device", "cpu"])).unwrap();
    wait(&job, 20);
    assert_eq!(texts(&job), ["args: --device cpu", "device: stub"]);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("done", Some(0)));
    assert_eq!(p.next, 2);
    // polling from `next` returns nothing new; from 1 returns the last line
    assert!(job.poll(p.next).lines.is_empty());
    let tail = job.poll(1).lines;
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].0, 1);
}

#[test]
fn stderr_is_flagged_and_a_failing_exit_is_an_error() {
    let (jobs, _d) = jobs();
    let job = jobs.start("eval", args(&[])).unwrap();
    wait(&job, 20);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("error", Some(3)));
    let err: Vec<_> = p
        .lines
        .iter()
        .filter(|(_, l)| l.err)
        .map(|(_, l)| &l.text)
        .collect();
    assert_eq!(err, ["something went wrong"]);
    assert!(p.lines.iter().any(|(_, l)| !l.err && l.text == "starting"));
}

#[test]
fn line_endings_and_a_missing_final_newline_are_handled() {
    let (jobs, _d) = jobs();
    let a = jobs.start("generate", args(&[])).unwrap();
    let b = jobs.start("export-gguf", args(&[])).unwrap();
    wait(&a, 20);
    wait(&b, 20);
    assert_eq!(texts(&a), ["no newline at the end"]);
    assert_eq!(texts(&b), ["crlf line"]);
}

#[test]
fn progress_lines_become_events_and_the_last_ones_are_kept() {
    let (jobs, dir) = jobs();
    let out = dir.join("m.gguf");
    let job = jobs
        .start("train", args(&["--out", out.to_str().unwrap()]))
        .unwrap();
    wait_for_line(&job, "waiting for");
    let p = job.poll(0);
    assert!(matches!(
        p.last_step,
        Some(Event::Step {
            step: 20,
            total: 100,
            ..
        })
    ));
    assert!(matches!(
        p.last_val,
        Some(Event::Val {
            epoch: 1,
            best: true,
            ..
        })
    ));
    assert_eq!(p.lines.iter().filter(|(_, l)| l.ev.is_some()).count(), 3);
    assert_eq!(p.state, "running");
    job.stop(true);
    wait(&job, 20);
}

#[test]
fn a_polite_stop_writes_the_stop_file_and_the_job_ends_stopped() {
    let (jobs, dir) = jobs();
    let out = dir.join("m.gguf");
    let job = jobs
        .start("train", args(&["--out", out.to_str().unwrap()]))
        .unwrap();
    wait_for_line(&job, "waiting for");
    assert!(job.stop(false));
    assert_eq!(job.word(), "stopping");
    wait(&job, 20);
    assert!(texts(&job).iter().any(|t| t == "stop file seen"));
    assert_eq!(job.word(), "stopped");
    // the stop file was removed so the next run does not stop at once
    assert!(!exists(&dir.join("m.stop")));
    assert!(!job.stop(false), "a finished job cannot be stopped");
}

#[test]
fn a_stale_stop_file_is_removed_before_a_run_starts() {
    let (jobs, dir) = jobs();
    let stale = dir.join("m.stop");
    std::fs::write(&stale, b"old").unwrap();
    let out = dir.join("m.gguf");
    let job = jobs
        .start("train", args(&["--out", out.to_str().unwrap()]))
        .unwrap();
    wait_for_line(&job, "waiting for");
    assert!(texts(&job)[0].contains("removed a stale"));
    assert!(!texts(&job).iter().any(|t| t == "stop file seen"));
    job.stop(true);
    wait(&job, 20);
}

#[test]
fn a_process_that_ignores_the_stop_file_is_killed_after_the_grace_period() {
    let (jobs, dir) = jobs();
    let model = dir.join("m.gguf");
    let job = jobs
        .start("run", args(&["--model", model.to_str().unwrap()]))
        .unwrap();
    wait_for_line(&job, "ignoring");
    let t0 = std::time::Instant::now();
    assert!(job.stop(false));
    wait(&job, 20);
    assert!(
        t0.elapsed() >= Duration::from_millis(300),
        "waited the grace period"
    );
    assert!(texts(&job)
        .iter()
        .any(|t| t.contains("Studio ended the process")));
    assert_eq!(job.word(), "stopped");
    assert!(job.poll(0).exit.is_some());
}

#[test]
fn a_command_without_a_stop_file_is_killed_at_once_and_force_skips_the_file() {
    let (jobs, _d) = jobs();
    let job = jobs.start("prepare", args(&["--out", "x"])).unwrap();
    wait_for_line(&job, "sleeping");
    assert!(job.stop(false));
    wait(&job, 20);
    assert_eq!(job.word(), "stopped");
    // force on a training command kills without writing the stop file
    let (jobs2, dir) = jobs_pair();
    let out = dir.join("f.gguf");
    let t = jobs2
        .start("train", args(&["--out", out.to_str().unwrap()]))
        .unwrap();
    wait_for_line(&t, "waiting for");
    assert!(t.stop(true));
    wait(&t, 20);
    assert!(!texts(&t).iter().any(|x| x == "stop file seen"));
}

fn jobs_pair() -> (Jobs, std::path::PathBuf) {
    jobs()
}

#[test]
fn an_unwritable_stop_file_falls_back_to_a_kill() {
    let (jobs, _d) = jobs();
    let job = jobs
        .start("train", args(&["--out", "/no/such/dir/m.gguf"]))
        .unwrap();
    wait_for_line(&job, "waiting for");
    assert!(job.stop(false));
    wait(&job, 20);
    assert_eq!(job.word(), "stopped");
}

#[test]
fn only_the_newest_lines_are_kept_and_indexes_stay_valid() {
    let (jobs, _d) = jobs();
    let job = jobs.start("tokenize", args(&[])).unwrap();
    wait(&job, 60);
    let p = job.poll(0);
    assert_eq!(p.next, 25_000);
    assert_eq!(p.lines.len(), MAX_LINES);
    assert_eq!(p.lines[0].0, 25_000 - MAX_LINES, "the first kept index");
    assert_eq!(p.lines.last().unwrap().1.text, "line 24999");
    // asking from a dropped index starts at the first kept one
    assert_eq!(job.poll(5).lines[0].0, 25_000 - MAX_LINES);
}

#[test]
fn too_many_running_jobs_are_refused() {
    let (jobs, _d) = jobs();
    let running: Vec<_> = (0..MAX_RUNNING)
        .map(|_| jobs.start("prepare", args(&[])).unwrap())
        .collect();
    let e = jobs.start("prepare", args(&[])).err().unwrap();
    assert!(e.contains("already running"), "{e}");
    for j in &running {
        j.stop(true);
    }
    for j in &running {
        wait(j, 20);
    }
    // room again
    let again = jobs.start("info", args(&[])).unwrap();
    wait(&again, 20);
}

#[test]
fn finished_jobs_beyond_the_limit_are_forgotten_oldest_first() {
    let (jobs, _d) = jobs();
    let mut ids = vec![];
    for _ in 0..(MAX_KEPT + 5) {
        let j = jobs.start("info", args(&[])).unwrap();
        wait(&j, 20);
        ids.push(j.id);
    }
    assert!(jobs.list().len() <= MAX_KEPT);
    assert!(jobs.get(ids[0]).is_none(), "the oldest is gone");
    assert!(jobs.get(*ids.last().unwrap()).is_some());
}

#[test]
fn start_errors_are_reported_not_panicked() {
    let missing = Jobs::new(Some("/definitely/not/here/ferrum-gpu".into()));
    assert!(missing
        .start("info", args(&[]))
        .err()
        .unwrap()
        .contains("no such file"));
    // refused by the allow-list before any process is looked for
    assert!(missing
        .start("sh", args(&[]))
        .err()
        .unwrap()
        .contains("not a command"));
    // a file that cannot be executed
    let (dir, _p) = stub();
    let plain = dir.join("plain");
    std::fs::write(&plain, b"not a program").unwrap();
    let j = Jobs::new(Some(plain));
    assert!(j
        .start("info", args(&[]))
        .err()
        .unwrap()
        .contains("cannot start"));
    assert!(j.get(1).is_none());
}

#[test]
fn set_cli_changes_the_binary_used() {
    let jobs = Jobs::new(None);
    let (_d, p) = stub();
    jobs.set_cli(Some(p));
    let job = jobs.start("info", args(&[])).unwrap();
    wait(&job, 20);
    assert_eq!(job.word(), "done");
}
