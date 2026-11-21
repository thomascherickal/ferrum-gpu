use super::*;

#[test]
fn the_close_watcher_tells_a_close_a_reset_and_silence_apart() {
    use std::io::ErrorKind;
    let run = |script: Vec<std::io::Result<usize>>| {
        let flag = AtomicBool::new(false);
        let done = AtomicBool::new(false);
        let mut script = script.into_iter();
        watch_with(|| script.next().unwrap_or(Ok(0)), &flag, &done);
        flag.load(Ordering::Relaxed)
    };
    // nothing yet, then pipelined bytes (neither is a close), then the peer closes
    assert!(run(vec![
        Err(ErrorKind::WouldBlock.into()),
        Err(ErrorKind::TimedOut.into()),
        Ok(3),
        Ok(0)
    ]));
    // a reset is a close too
    assert!(run(vec![Err(ErrorKind::ConnectionReset.into())]));
    // the handler finishing first leaves the flag alone
    let flag = AtomicBool::new(false);
    let done = AtomicBool::new(true);
    watch_with(|| panic!("never polled"), &flag, &done);
    assert!(!flag.load(Ordering::Relaxed));
}
