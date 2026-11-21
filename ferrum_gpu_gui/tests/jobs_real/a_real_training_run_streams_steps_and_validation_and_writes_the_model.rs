use super::*;

#[test]
fn a_real_training_run_streams_steps_and_validation_and_writes_the_model() {
    let dir = work_dir("full");
    let jobs = Jobs::new();
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
    let dir = work_dir("stop");
    let jobs = Jobs::new();
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
    assert!(job.stop(), "the stop was accepted");
    wait(&job, 60);
    let p = job.poll(0);
    assert_eq!(p.state, "stopped", "{:?}", p.lines);
    assert_eq!(p.exit, Some(0), "the command ended itself: {:?}", p.lines);
    assert!(
        p.lines.iter().any(|(_, l)| l.text.contains("cancelled")),
        "{:?}",
        p.lines
    );
    assert!(dir.join("m.gguf").is_file(), "the partial model was kept");
}

#[test]
fn two_trainings_cannot_open_the_device_at_once_and_the_second_can_start_after_the_first() {
    let dir = work_dir("exclusive");
    let jobs = Jobs::new();
    let first = jobs
        .start("train", args(&dir, &["--steps", "200000"]))
        .unwrap();
    let e = jobs
        .start("train", args(&dir, &["--steps", "5"]))
        .err()
        .expect("refused while the first runs");
    assert!(e.contains("using the device"), "{e}");
    first.stop();
    wait(&first, 60);
    let second = jobs.start("train", args(&dir, &["--steps", "5"])).unwrap();
    wait(&second, 60);
    assert_eq!(second.word(), "done", "{:?}", second.poll(0).lines);
}
