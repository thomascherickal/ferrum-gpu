use super::*;

#[test]
fn a_failed_start_says_why_and_leaves_the_slot_free() {
    let slot = ServeSlot::new();
    for (bad, needle) in [
        ("--port 0".to_string(), "--model"),
        (
            "--model /no/such.gguf --port 0 --device cpu".to_string(),
            "no/such.gguf",
        ),
        ("--model x --device tpu".to_string(), "tpu"),
        ("--model x --nope".to_string(), "--nope"),
        (
            "--model x --host 0.0.0.0".to_string(),
            "not a loopback address",
        ),
    ] {
        let e = slot.start(&args(&bad)).unwrap_err();
        assert!(e.contains(needle), "{bad}: {e}");
        let st = slot.status();
        assert!(!st.running);
        assert!(
            st.lines.last().unwrap().starts_with("could not start: "),
            "{:?}",
            st.lines
        );
    }
    // and a good start still works afterwards
    let m = model();
    slot.start(&args(&format!(
        "--model {} --port 0 --device cpu",
        m.display()
    )))
    .unwrap();
    slot.stop();
}

#[test]
fn the_kept_lines_are_bounded() {
    let lines = Arc::new(Mutex::new(vec![]));
    let mut w = Lines(lines.clone(), vec![]);
    for i in 0..(MAX_LINES + 30) {
        writeln!(w, "line {i}").unwrap();
    }
    write!(w, "unfinished").unwrap();
    w.flush().unwrap();
    let v = lock(&lines);
    assert_eq!(v.len(), MAX_LINES);
    assert_eq!(v.last().unwrap(), &format!("line {}", MAX_LINES + 29));
    assert_eq!(v[0], "line 30", "the oldest were dropped");
}
