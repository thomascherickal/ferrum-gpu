use ferrum_wgpu::bench_util::time_ms;
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::matmul::Matmul;

/// Milliseconds of one GPU variant: compute only, plus readback, plus upload, plus both.
pub struct Timings {
    pub compute: f64,
    pub with_readback: f64,
    pub with_upload: f64,
    pub with_all: f64,
}

pub fn measure(
    gpu: &Gpu,
    mm: &Matmul,
    (a, b): (&[f32], &[f32]),
    mnk: (usize, usize, usize),
    reps: usize,
) -> Timings {
    let (m, k, n) = mnk;
    let (ab, bb, cb) = (
        gpu.upload(a).unwrap(),
        gpu.upload(b).unwrap(),
        gpu.zeros(m * n).unwrap(),
    );
    let compute = time_ms(reps, || {
        mm.submit(gpu, &ab, &bb, &cb, (m, k, n)).unwrap();
        gpu.wait().unwrap();
    });
    let with_readback = time_ms(reps, || {
        mm.submit(gpu, &ab, &bb, &cb, (m, k, n)).unwrap();
        std::hint::black_box(gpu.read(&cb).unwrap());
    });
    let with_upload = time_ms(reps, || {
        let (ab, bb) = (gpu.upload(a).unwrap(), gpu.upload(b).unwrap());
        mm.submit(gpu, &ab, &bb, &cb, (m, k, n)).unwrap();
        gpu.wait().unwrap();
    });
    let with_all = time_ms(reps, || {
        let (ab, bb) = (gpu.upload(a).unwrap(), gpu.upload(b).unwrap());
        mm.submit(gpu, &ab, &bb, &cb, (m, k, n)).unwrap();
        std::hint::black_box(gpu.read(&cb).unwrap());
    });
    Timings {
        compute,
        with_readback,
        with_upload,
        with_all,
    }
}
