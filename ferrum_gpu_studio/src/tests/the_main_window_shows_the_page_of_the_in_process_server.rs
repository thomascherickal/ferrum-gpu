use super::*;

#[test]
fn the_main_window_shows_the_page_of_the_in_process_server() {
    let server = cpu_server();
    let url = server.url();
    let mut app = configure(mock_builder(), server)
        .build(mock_context(noop_assets()))
        .expect("the app builds");
    // Tauri runs the setup hook (which creates the window) on the first event-loop iteration
    #[allow(deprecated)]
    app.run_iteration(|_, _| {});
    let w = app
        .get_webview_window("main")
        .expect("the main window exists");
    assert_eq!(w.url().unwrap().as_str(), url);
    // (the mock runtime does not keep window titles, so TITLE is only checked to be a real name)
    assert!(TITLE.contains("Studio") && !TITLE.is_empty());
    let studio = app.state::<Studio>();
    assert_eq!(studio.url, url);
    assert!(studio.running());
    studio.stop();
}

#[test]
fn that_page_is_the_studio_with_its_policy() {
    let server = cpu_server();
    let port = server.addr.port();
    let page = get(port, "/");
    assert!(
        page.starts_with("HTTP/1.1 200"),
        "{}",
        &page[..40.min(page.len())]
    );
    assert!(
        page.contains("Content-Security-Policy"),
        "the policy is on the page"
    );
    assert!(
        page.contains("Datasets") && page.contains("ui-mode"),
        "the Studio page"
    );
    server.stop();
}

#[test]
fn the_server_refuses_a_foreign_host_so_the_window_is_the_only_client() {
    let server = cpu_server();
    let port = server.addr.port();
    let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    write!(
        c,
        "GET / HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    c.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 403"), "{out}");
    server.stop();
}

#[test]
fn stopping_ends_the_server_and_a_second_stop_does_nothing() {
    let server = cpu_server();
    let port = server.addr.port();
    let studio = Studio::new(server);
    assert!(studio.running());
    studio.stop();
    assert!(!studio.running());
    studio.stop();
    assert!(
        TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}").parse().unwrap(),
            Duration::from_millis(500)
        )
        .is_err(),
        "nothing listens any more"
    );
}
