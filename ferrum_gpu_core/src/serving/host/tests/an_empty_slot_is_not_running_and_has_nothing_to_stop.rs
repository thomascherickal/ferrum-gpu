use super::*;

#[test]
fn an_empty_slot_is_not_running_and_has_nothing_to_stop() {
    let s = ServeSlot::new();
    let st = s.status();
    assert!(!st.running && st.addr.is_none() && st.seconds.is_none() && st.lines.is_empty());
    assert!(!st.has_token);
    assert!(!s.stop());
}

#[test]
fn a_server_starts_answers_and_stops_and_a_second_start_is_refused_meanwhile() {
    let m = model();
    let slot = ServeSlot::new();
    let st = slot
        .start(&args(&format!(
            "--model {} --port 0 --device cpu --threads 1",
            m.display()
        )))
        .unwrap();
    assert!(st.running && !st.has_token);
    let addr = st.addr.clone().unwrap();
    assert!(
        st.lines.iter().any(|l| l.starts_with("device: ")),
        "{:?}",
        st.lines
    );
    assert!(
        st.lines
            .iter()
            .any(|l| l.contains("ferrum-gpu server: http://")),
        "{:?}",
        st.lines
    );
    // it really serves
    let health = get(&addr, "/health", None);
    assert!(
        health.starts_with("HTTP/1.1 200") && health.contains("\"ok\""),
        "{health}"
    );
    let info = get(&addr, "/v1/info", None);
    assert!(info.contains("\"arch\"") || info.contains("gpt2"), "{info}");
    // one at a time
    let e = slot
        .start(&args(&format!(
            "--model {} --port 0 --device cpu",
            m.display()
        )))
        .unwrap_err();
    assert!(e.contains("already running"), "{e}");
    assert!(slot.status().seconds.unwrap() >= 0.0);
    // stopping ends it
    assert!(slot.stop());
    let after = slot.status();
    assert!(!after.running && after.lines.last().unwrap() == "server stopped");
    assert!(
        TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(500)).is_err()
    );
    assert!(!slot.stop(), "a second stop does nothing");
}

#[test]
fn a_token_protects_the_server_and_is_reported() {
    let m = model();
    let slot = ServeSlot::new();
    let st = slot
        .start(&args(&format!(
            "--model {} --port 0 --device cpu --token s3cret",
            m.display()
        )))
        .unwrap();
    assert!(st.has_token);
    assert!(st.lines.iter().any(|l| l.starts_with("auth:")));
    let addr = st.addr.unwrap();
    assert!(get(&addr, "/v1/info", None).starts_with("HTTP/1.1 401"));
    assert!(get(&addr, "/v1/info", Some("s3cret")).starts_with("HTTP/1.1 200"));
    assert!(
        get(&addr, "/health", None).starts_with("HTTP/1.1 200"),
        "health is open"
    );
    slot.stop();
}
