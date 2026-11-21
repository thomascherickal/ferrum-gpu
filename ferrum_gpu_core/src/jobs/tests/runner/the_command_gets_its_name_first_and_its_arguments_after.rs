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
