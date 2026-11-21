use super::*;

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
