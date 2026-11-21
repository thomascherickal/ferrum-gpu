use super::*;

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
