use ferrum_ref::matmul::rand_vec;
use ferrum_wgpu::bench_util::time_ms;
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::tensor::GpuBuf;

/// The "large" training shape: batch, tokens, dim, hidden, vocab, heads, layers.
pub(crate) struct Dims {
    pub b: usize,
    pub t: usize,
    pub c: usize,
    pub h: usize,
    pub v: usize,
    pub heads: usize,
    pub layers: usize,
}

impl Dims {
    pub(crate) fn rows(&self) -> usize {
        self.b * self.t
    }
}

/// The device, the shape, and the timed rows (name, ms each, runs per step) collected so far.
pub(crate) struct Rig {
    pub g: Gpu,
    pub d: Dims,
    rows: Vec<(String, f64, usize)>,
}

impl Rig {
    pub(crate) fn new(g: Gpu, d: Dims) -> Rig {
        Rig {
            g,
            d,
            rows: Vec::new(),
        }
    }

    /// A device buffer of `n` random values.
    pub(crate) fn buf(&self, n: usize) -> GpuBuf {
        self.g.upload(&rand_vec(n, n as u64)).unwrap()
    }

    pub(crate) fn zeros(&self, n: usize) -> GpuBuf {
        self.g.zeros(n).unwrap()
    }

    /// Time `op` (submit + wait) and record it as run `per_step` times in a training step.
    pub(crate) fn record(&mut self, name: &str, per_step: usize, mut op: impl FnMut(&Gpu)) {
        let g = &self.g;
        let ms = time_ms(7, || {
            op(g);
            g.wait().unwrap();
        });
        self.rows.push((name.to_string(), ms, per_step));
    }

    pub(crate) fn print_table(&self) {
        println!("| kernel | ms each | per step | ms per step | share |");
        println!("|---|---|---|---|---|");
        let total: f64 = self.rows.iter().map(|r| r.1 * r.2 as f64).sum();
        for (name, t_ms, per) in &self.rows {
            let step = t_ms * *per as f64;
            println!(
                "| {name} | {t_ms:.3} | {per} | {step:.1} | {:.0}% |",
                100.0 * step / total
            );
        }
        println!("| **estimated sum** | | | **{total:.0}** | |");
    }
}
