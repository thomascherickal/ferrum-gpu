use super::support::{args, block_until_cancelled, jobs_with, texts, wait, wait_for_line};
use crate::jobs::progress::Event;
use crate::jobs::runner::LineSink;
use crate::jobs::{Job, Jobs, MAX_KEPT, MAX_LINES, MAX_RUNNING};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[test]
fn the_command_gets_its_name_first_and_its_arguments_after() {
    let seen = Arc::new(Mutex::new(vec![]));
    let s2 = seen.clone();
    let jobs = jobs_with(move |argv, _| {
        s2.lock().unwrap().extend(argv.iter().cloned());
        Ok(())
    });
    let job = jobs
        .start("eval", args(&["--model", "m.gguf", "--ctx", "8"]))
        .unwrap();
    wait(&job, 20);
    assert_eq!(
        *seen.lock().unwrap(),
        ["eval", "--model", "m.gguf", "--ctx", "8"]
    );
}

#[test]
fn output_is_streamed_line_by_line_and_success_is_done() {
    let jobs = jobs_with(|_, out| {
        writeln!(out, "first").unwrap();
        write!(out, "sec").unwrap();
        writeln!(out, "ond").unwrap();
        Ok(())
    });
    let job = jobs.start("prepare", args(&[])).unwrap();
    wait(&job, 20);
    assert_eq!(texts(&job), ["first", "second"]);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("done", Some(0)));
    assert_eq!(p.next, 2);
    assert!(job.poll(p.next).lines.is_empty());
    let tail = job.poll(1).lines;
    assert_eq!((tail.len(), tail[0].0), (1, 1));
}

#[test]
fn an_error_is_an_error_line_and_exit_status_two() {
    let jobs = jobs_with(|_, out| {
        writeln!(out, "starting").unwrap();
        Err("--data is required".into())
    });
    let job = jobs.start("train", args(&[])).unwrap();
    wait(&job, 20);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("error", Some(2)));
    assert_eq!(p.lines.len(), 2);
    assert!(!p.lines[0].1.err && p.lines[1].1.err);
    assert_eq!(p.lines[1].1.text, "--data is required");
}

#[test]
fn a_panic_ends_the_job_not_the_process() {
    let jobs = jobs_with(|_, out| {
        writeln!(out, "about to fail").unwrap();
        if true {
            panic!("kernel exploded");
        }
        Ok(())
    });
    let job = jobs.start("generate", args(&[])).unwrap();
    wait(&job, 20);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("error", Some(101)));
    let last = &p.lines.last().unwrap().1;
    assert!(
        last.err && last.text == "panicked: kernel exploded",
        "{last:?}"
    );
    // a String payload and an unknown payload are described too
    let j2 = jobs_with(|_, _| -> Result<(), String> { panic!("{}", String::from("formatted 7")) });
    let b = j2.start("eval", args(&[])).unwrap();
    wait(&b, 20);
    assert_eq!(
        b.poll(0).lines.last().unwrap().1.text,
        "panicked: formatted 7"
    );
    let j3 = jobs_with(|_, _| -> Result<(), String> { std::panic::panic_any(42_u8) });
    let c = j3.start("eval", args(&[])).unwrap();
    wait(&c, 20);
    assert_eq!(
        c.poll(0).lines.last().unwrap().1.text,
        "panicked: unknown panic"
    );
}

#[test]
fn progress_lines_become_events_and_the_last_ones_are_kept() {
    let jobs = jobs_with(|_, out| {
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
        writeln!(out, "not progress").unwrap();
        Ok(())
    });
    let job = jobs.start("train", args(&[])).unwrap();
    wait(&job, 20);
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
}

#[test]
fn stop_raises_the_cancel_flag_the_command_polls_and_the_job_ends_stopped() {
    let jobs = jobs_with(|_, out| {
        writeln!(out, "working").unwrap();
        block_until_cancelled();
        writeln!(out, "cancel seen").unwrap();
        Ok(())
    });
    let job = jobs.start("train", args(&[])).unwrap();
    wait_for_line(&job, "working");
    assert_eq!(job.word(), "running");
    assert!(job.stop());
    assert!(matches!(job.word(), "stopping" | "stopped"));
    wait(&job, 20);
    assert!(texts(&job).contains(&"cancel seen".to_string()));
    assert_eq!(job.word(), "stopped");
    assert!(!job.stop(), "a finished job cannot be stopped");
}

#[test]
fn one_jobs_cancel_flag_does_not_reach_another() {
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = n.clone();
    let jobs = jobs_with(move |argv, out| {
        if argv[0] == "prepare" {
            block_until_cancelled();
            writeln!(out, "prepare cancelled").unwrap();
        } else {
            n2.fetch_add(1, Ordering::SeqCst);
            assert!(!crate::cancel::requested(), "this job was not stopped");
            writeln!(out, "eval fine").unwrap();
        }
        Ok(())
    });
    let a = jobs.start("prepare", args(&[])).unwrap();
    let b = jobs.start("eval", args(&[])).unwrap();
    wait(&b, 20);
    assert_eq!(b.word(), "done");
    a.stop();
    wait(&a, 20);
    assert_eq!(n.load(Ordering::SeqCst), 1);
}

#[test]
fn only_the_newest_lines_are_kept_and_indexes_stay_valid() {
    let jobs = jobs_with(|_, out| {
        for i in 0..25_000 {
            writeln!(out, "line {i}").unwrap();
        }
        Ok(())
    });
    let job = jobs.start("tokenize", args(&[])).unwrap();
    wait(&job, 60);
    let p = job.poll(0);
    assert_eq!(p.next, 25_000);
    assert_eq!(p.lines.len(), MAX_LINES);
    assert_eq!(p.lines[0].0, 25_000 - MAX_LINES, "the first kept index");
    assert_eq!(p.lines.last().unwrap().1.text, "line 24999");
    assert_eq!(job.poll(5).lines[0].0, 25_000 - MAX_LINES);
}

#[test]
fn commands_that_open_a_device_take_turns_but_the_others_run_beside_them() {
    let jobs = jobs_with(|_, _| {
        block_until_cancelled();
        Ok(())
    });
    let train = jobs.start("train", args(&[])).unwrap();
    let e = jobs.start("generate", args(&[])).err().unwrap();
    assert!(e.contains("using the device") && e.contains("train"), "{e}");
    let e = jobs.start("info", args(&[])).err().unwrap();
    assert!(e.contains("job 1"), "{e}");
    // these do not open a device
    let others: Vec<_> = ["prepare", "tokenize", "export-gguf"]
        .iter()
        .map(|c| jobs.start(c, args(&[])).unwrap())
        .collect::<Vec<_>>();
    assert!(jobs
        .start("export-ollama", args(&[]))
        .err()
        .unwrap()
        .contains("already running"));
    for j in std::iter::once(&train).chain(others.iter()) {
        j.stop();
        wait(j, 20);
    }
    // the device is free again
    let again = jobs.start("generate", args(&[])).unwrap();
    again.stop();
    wait(&again, 20);
}

#[test]
fn too_many_running_jobs_are_refused_and_room_comes_back() {
    let jobs = jobs_with(|_, _| {
        block_until_cancelled();
        Ok(())
    });
    let running: Vec<_> = (0..MAX_RUNNING)
        .map(|_| jobs.start("prepare", args(&[])).unwrap())
        .collect();
    let e = jobs.start("prepare", args(&[])).err().unwrap();
    assert!(e.contains("already running"), "{e}");
    for j in &running {
        j.stop();
        wait(j, 20);
    }
    let again = jobs.start("prepare", args(&[])).unwrap();
    again.stop();
    wait(&again, 20);
}

#[test]
fn finished_jobs_beyond_the_limit_are_forgotten_oldest_first() {
    let jobs = jobs_with(|_, _| Ok(()));
    let mut ids = vec![];
    for _ in 0..(MAX_KEPT + 5) {
        let j = jobs.start("eval", args(&[])).unwrap();
        wait(&j, 20);
        ids.push(j.id);
    }
    assert!(jobs.list().len() <= MAX_KEPT);
    assert!(jobs.get(ids[0]).is_none(), "the oldest is gone");
    assert!(jobs.get(*ids.last().unwrap()).is_some());
}

#[test]
fn a_request_the_allow_list_refuses_never_reaches_the_runner() {
    let called = Arc::new(AtomicUsize::new(0));
    let c2 = called.clone();
    let jobs = jobs_with(move |_, _| {
        c2.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    assert!(jobs
        .start("sh", args(&["-c", "true"]))
        .err()
        .unwrap()
        .contains("not a command"));
    assert!(jobs
        .start("export-ollama", args(&["--bin", "/bin/sh"]))
        .is_err());
    assert_eq!(called.load(Ordering::SeqCst), 0);
    assert!(jobs.get(1).is_none());
}

#[test]
fn the_line_sink_handles_crlf_chunks_and_an_unfinished_last_line() {
    let job = Arc::new(Job::new(1, "eval", vec![]));
    let mut sink = LineSink::new(job.clone());
    sink.write_all(b"one\r\ntw").unwrap();
    sink.write_all(b"o\n\nthree").unwrap();
    sink.flush().unwrap();
    assert_eq!(texts(&job), ["one", "two", ""]);
    sink.finish();
    assert_eq!(texts(&job), ["one", "two", "", "three"]);
    sink.finish();
    assert_eq!(texts(&job).len(), 4, "finishing twice adds nothing");
}

#[test]
fn the_real_info_command_runs_in_process() {
    let jobs = Jobs::new();
    let job = jobs.start("info", args(&["--device", "cpu"])).unwrap();
    wait(&job, 120);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("done", Some(0)), "{:?}", p.lines);
    assert!(
        texts(&job).iter().any(|t| t.to_lowercase().contains("cpu")),
        "{:?}",
        texts(&job)
    );
}

#[test]
fn the_real_command_reports_its_own_errors_as_lines() {
    let jobs = Jobs::default();
    let job = jobs.start("eval", args(&[])).unwrap();
    wait(&job, 60);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("error", Some(2)));
    assert!(p.lines.iter().any(|(_, l)| l.err), "{:?}", p.lines);
}
