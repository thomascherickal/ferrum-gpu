use super::*;

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
