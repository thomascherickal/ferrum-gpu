use crate::decode::{decode_ms, mem_available};
use crate::row;
use crate::shapes::Shape;
use crate::PRECISIONS;
use ferrum_ref::llama::Llama;
use ferrum_wgpu::bench_util::mib;
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::llama::{plan, GpuLlama};
use ferrum_wgpu::llama_train::param_sizes;

pub fn shape_rows(gpu: &Gpu, s: &Shape, only_prec: Option<&str>) {
    let info = gpu.info();
    let n_params: usize = param_sizes(&s.cfg, s.biases).iter().sum();
    let params = format!("{:.0}M", n_params as f64 / 1e6);
    let fits: Vec<bool> = PRECISIONS
        .iter()
        .map(|(_, q)| plan(&s.cfg, *q).check(info.max_binding, None).is_ok())
        .collect();
    // random f32 weights are built on the host first: keep 1.5x their size free so a big
    // shape cannot push the machine into swap or the OOM killer
    let need = n_params as u64 * 4 * 3 / 2;
    let free = mem_available();
    if fits.iter().any(|&f| f) && free.is_some_and(|a| a < need) {
        let note = format!(
            "skipped: needs {} MiB of free host RAM for the f32 weights, {} MiB available",
            mib(need),
            mib(free.unwrap_or(0))
        );
        return row::print(s.name, &params, "all", "-", None, &note);
    }
    // weights are built once per shape, and only if some precision fits
    let w = fits
        .iter()
        .any(|&f| f)
        .then(|| Llama::random(s.cfg, 1, s.biases, s.tied));
    for ((label, q), fit) in PRECISIONS.iter().zip(&fits) {
        if only_prec.is_some_and(|o| o != *label) {
            continue;
        }
        let p = plan(&s.cfg, *q);
        let total = mib(p.total());
        // an integrated GPU's memory is the machine's RAM: its buffers count against it too
        let shared = format!("{:?}", info.kind) == "Integrated"
            && free.is_some_and(|a| a < need + p.total());
        let (ms, note) = if shared && *fit {
            let note = format!("skipped: integrated GPU shares RAM; weights on the host plus the plan exceed the {} MiB available", mib(free.unwrap_or(0)));
            (None, note)
        } else if !fit {
            let why = p
                .check(info.max_binding, None)
                .err()
                .map_or(String::new(), |e| e.to_string());
            (None, format!("refused: {why}"))
        } else {
            let w = w.as_ref().expect("weights exist when something fits");
            match GpuLlama::new(gpu, w, *q, None) {
                Err(e) => (None, format!("failed to load: {e}")),
                Ok(mut m) => match decode_ms(&mut m, s.cfg.vocab) {
                    Ok(ms) => (Some(ms), String::new()),
                    Err(e) => (None, format!("failed: {e}")),
                },
            }
        };
        row::print(s.name, &params, label, &total, ms, &note);
    }
}
