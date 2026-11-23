use super::*;

#[test]
fn an_eval_result_line_is_kept_as_the_latest_eval() {
    let jobs = jobs_with(|_, out| {
        writeln!(out, "tokens: 4096 loss: 2.3456 perplexity: 10.442").unwrap();
        Ok(())
    });
    let job = jobs.start("eval", args(&[])).unwrap();
    wait(&job, 20);
    assert!(matches!(
        job.poll(0).last_eval,
        Some(Event::Eval { tokens: 4096, .. })
    ));
}
