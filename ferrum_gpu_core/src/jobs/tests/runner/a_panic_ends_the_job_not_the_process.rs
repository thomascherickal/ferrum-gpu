use super::*;

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
