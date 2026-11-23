use ferrum_wgpu::llama::GpuLlama;
use std::time::Instant;

pub(crate) const NEW_TOKENS: usize = 16;
const PROMPT: usize = 4;

/// Available system RAM in bytes; `None` if the platform does not say.
pub(crate) fn mem_available() -> Option<u64> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    Some(sys.available_memory()).filter(|&b| b > 0)
}

fn run(m: &mut GpuLlama, toks: &[usize]) -> Result<f64, String> {
    for &t in &toks[..PROMPT] {
        m.feed(t).map_err(|e| e.to_string())?;
    }
    let t0 = Instant::now();
    for &t in &toks[PROMPT..] {
        std::hint::black_box(m.feed(t).map_err(|e| e.to_string())?);
    }
    Ok(t0.elapsed().as_secs_f64() * 1e3 / NEW_TOKENS as f64)
}

/// Milliseconds per token after a 4-token prompt. The first pass compiles the pipelines; the
/// second is the one reported.
pub(crate) fn decode_ms(m: &mut GpuLlama, vocab: usize) -> Result<f64, String> {
    let toks: Vec<usize> = (0..PROMPT + NEW_TOKENS)
        .map(|i| (i * 7 + 3) % vocab)
        .collect();
    let warm = run(m, &toks);
    m.reset();
    warm.and_then(|_| run(m, &toks))
}
