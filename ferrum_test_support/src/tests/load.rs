use crate::load::{ask, drive, percentile, summary};
use std::io::{Read, Write};
use std::net::TcpListener;

#[test]
fn percentiles_use_the_nearest_rank() {
    let v: Vec<f64> = (1..=100).map(f64::from).collect();
    assert_eq!(percentile(&v, 50.0), 50.0);
    assert_eq!(percentile(&v, 99.0), 99.0);
    assert_eq!(percentile(&v, 0.0), 1.0);
    assert_eq!(percentile(&[7.0], 95.0), 7.0);
}

#[test]
fn the_driver_counts_statuses_and_a_reply_without_a_status_is_zero() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        for (n, c) in l.incoming().take(5).enumerate() {
            let mut c = c.unwrap();
            let mut b = [0u8; 64];
            let _ = c.read(&mut b).unwrap();
            let reply: &[u8] = match n {
                0 => b"junk",
                n if n % 2 == 1 => b"HTTP/1.1 200 OK\r\n\r\nyes",
                _ => b"HTTP/1.1 503 X\r\n\r\nbusy",
            };
            c.write_all(reply).unwrap();
        }
    });
    assert_eq!(ask(addr, b"x").status, 0);
    let (answers, wall) = drive(addr, 2, 2, |i| format!("r{i}").into_bytes());
    server.join().unwrap();
    let line = summary("t", 2, &answers, wall);
    assert!(
        line.starts_with("LOAD name=t clients=2 requests=4 ok=2 refused=2 "),
        "{line}"
    );
    assert!(answers.iter().any(|a| a.body == "busy"));
}
