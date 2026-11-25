//! A small HTTP load driver for the servers' load tests: many clients on plain TCP (std threads),
//! one request per connection, every request timed. `summary` prints one `LOAD ...` line that
//! `scripts/load_profile.py` reads into system-profile.md.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

/// One answered request.
#[derive(Debug)]
pub struct Answer {
    /// The status code (0 when the answer had no status line).
    pub status: u16,
    /// The body, as text.
    pub body: String,
    /// From connecting to the last byte read.
    pub took: Duration,
}

/// Send `raw` on a new connection and read the whole answer.
pub fn ask(addr: SocketAddr, raw: &[u8]) -> Answer {
    let t0 = Instant::now();
    let mut c = TcpStream::connect(addr).expect("connect to the server");
    c.write_all(raw).expect("send the request");
    let mut out = Vec::new();
    c.read_to_end(&mut out).expect("read the answer");
    let text = String::from_utf8_lossy(&out);
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status = head.split(' ').nth(1).and_then(|s| s.parse().ok());
    Answer {
        status: status.unwrap_or(0),
        body: body.to_string(),
        took: t0.elapsed(),
    }
}

/// `clients` threads, each sending `each` requests built by `raw(request number)`, all at once.
/// Returns every answer and the wall time of the whole run.
pub fn drive(
    addr: SocketAddr,
    clients: usize,
    each: usize,
    raw: impl Fn(usize) -> Vec<u8> + Sync,
) -> (Vec<Answer>, Duration) {
    let t0 = Instant::now();
    let answers = std::thread::scope(|s| {
        let raw = &raw;
        let threads: Vec<_> = (0..clients)
            .map(|c| {
                s.spawn(move || {
                    (0..each)
                        .map(|i| ask(addr, &raw(c * each + i)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        threads
            .into_iter()
            .flat_map(|t| t.join().unwrap_or_default())
            .collect()
    });
    (answers, t0.elapsed())
}

/// Nearest-rank percentile `p` (0 to 100) of `sorted` (ascending, not empty).
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// The one-line summary of a run: counts, throughput and latency percentiles in milliseconds.
pub fn summary(name: &str, clients: usize, answers: &[Answer], wall: Duration) -> String {
    let mut ms: Vec<f64> = answers.iter().map(|a| a.took.as_secs_f64() * 1e3).collect();
    ms.sort_by(f64::total_cmp);
    let count = |s: u16| answers.iter().filter(|a| a.status == s).count();
    let secs = wall.as_secs_f64();
    format!(
        "LOAD name={name} clients={clients} requests={} ok={} refused={} seconds={secs:.2} rps={:.0} p50_ms={:.1} p95_ms={:.1} p99_ms={:.1} max_ms={:.1}",
        answers.len(),
        count(200),
        count(503),
        answers.len() as f64 / secs,
        percentile(&ms, 50.0),
        percentile(&ms, 95.0),
        percentile(&ms, 99.0),
        percentile(&ms, 100.0),
    )
}
