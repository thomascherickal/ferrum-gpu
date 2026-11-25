use super::*;
use std::cell::Cell;
use std::time::Instant;

/// Gives one byte of `bytes` per read; every read moves the fake clock on by one second.
struct Dribble<'a>(&'a [u8], &'a Cell<Instant>);

impl Read for Dribble<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.1.set(self.1.get() + Duration::from_secs(1));
        let Some((first, rest)) = self.0.split_first() else {
            return Ok(0);
        };
        (buf[0], self.0) = (*first, rest);
        Ok(1)
    }
}

fn read_on_fake_clock(bytes: &[u8]) -> Result<Request, HttpError> {
    let clock = Cell::new(Instant::now());
    let now = || clock.get();
    read_request_within(
        &mut Dribble(bytes, &clock),
        1 << 21,
        Budget::default(),
        &now,
    )
}

#[test]
fn the_head_and_the_body_each_get_a_total_budget_not_a_per_read_one() {
    let short = b"GET / HTTP/1.1\r\n\r\n"; // 18 one-second reads: past the 10 s head budget
    assert_eq!(read_on_fake_clock(short).unwrap_err().status, 408);
    assert_eq!(read_on_fake_clock(b"GET").unwrap_err().status, 400);
    let mut post = b"POST / HTTP/1.1\r\nContent-Length: 25\r\n\r\n".to_vec();
    let clock = Cell::new(Instant::now());
    let fast = || clock.get(); // the head is read before the clock moves: only the body dribbles
    let mut head = std::io::Cursor::new(post.clone());
    assert!(read_request_within(&mut head, 64, Budget::default(), &fast).is_err());
    post.extend([b'x'; 25]); // 25 one-second reads: past 10 s + 10 s for one started MiB
    let mut r = std::io::Cursor::new(post[..39].to_vec()).chain(Dribble(&post[39..], &clock));
    let e = read_request_within(&mut r, 64, Budget::default(), &fast).unwrap_err();
    assert_eq!(
        (e.status, e.message.as_str()),
        (408, "timed out reading the body")
    );
    let b = Budget::default();
    assert_eq!(b.body(0), Duration::from_secs(10));
    assert_eq!(b.body((1 << 20) + 1), Duration::from_secs(30));
    assert!(b.body(usize::MAX) > Duration::from_secs(u64::from(u32::MAX)));
    let huge = Budget {
        per_mib: Duration::MAX,
        ..b
    };
    assert_eq!(huge.body(1), Duration::MAX);
}

#[test]
fn a_real_dribbling_client_is_cut_off_and_others_are_served_meanwhile() {
    let mut opts = ServeOptions::new(1024, false);
    opts.budget.head = Duration::from_millis(300);
    let (port, stop, t) = serve_work_with(opts, |_| Response::new(200, "text/plain", "ok"));
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut slow = TcpStream::connect(addr).unwrap();
    let mut writer = slow.try_clone().unwrap();
    let t0 = Instant::now();
    let dribbler = std::thread::spawn(move || {
        // one byte every 50 ms: each read is quick, the whole head never ends
        let _ = writer.write_all(b"GET / HTTP/1.1\r\nX-Slow: ");
        while t0.elapsed() < Duration::from_secs(5) && writer.write_all(b"a").is_ok() {
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    let get = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
    assert_eq!(ferrum_test_support::load::ask(addr, get).status, 200);
    let mut answer = String::new();
    let _ = slow.read_to_string(&mut answer);
    assert!(answer.starts_with("HTTP/1.1 408 "), "{answer}");
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    dribbler.join().unwrap();
    assert_eq!(ferrum_test_support::load::ask(addr, get).status, 200);
    stop.store(true, Ordering::Relaxed);
    t.join().unwrap();
}
