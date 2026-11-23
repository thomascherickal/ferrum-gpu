use super::*;

#[test]
fn a_thread_that_cannot_start_is_reported() {
    let e = started(Err(std::io::Error::other("out of threads"))).unwrap_err();
    assert_eq!(e, "cannot start a thread for the job: out of threads");
    assert!(started(std::thread::Builder::new().spawn(|| {})).is_ok());
}
