use super::*;

#[test]
fn the_real_command_reports_its_own_errors_as_lines() {
    let jobs = Jobs::default();
    let job = jobs.start("eval", args(&[])).unwrap();
    wait(&job, 60);
    let p = job.poll(0);
    assert_eq!((p.state, p.exit), ("error", Some(2)));
    assert!(p.lines.iter().any(|(_, l)| l.err), "{:?}", p.lines);
}
