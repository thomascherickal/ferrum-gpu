use super::*;

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
