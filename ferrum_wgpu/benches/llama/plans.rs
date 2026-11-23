use crate::shapes::{params, shape, sizes};
use ferrum_wgpu::bench_util::mib;
use ferrum_wgpu::budget::BufferPlan;
use ferrum_wgpu::device::Gpu;
use ferrum_wgpu::llama::plan;
use ferrum_wgpu::llama_train::{activation_plan, param_sizes, static_plan_for};
use ferrum_wgpu::ops_quant::QKind;

pub(crate) fn table(gpu: &Gpu) {
    println!("\n### Memory plans (MiB) against this adapter's binding limit\n");
    println!("| model | params | infer f32 | infer Q8_0 | infer Q4_0 | fine-tune f32 (batch 1 x 128) | verdict |");
    println!("|---|---|---|---|---|---|---|");
    let s = sizes();
    let refs = [
        ("small (run above)", s[0].1, false),
        ("medium (run above)", s[1].1, false),
        ("large (run above)", s[2].1, false),
        (
            "SmolLM-135M shape (plan only)",
            shape((49152, 2048, 576), (9, 3, 64), (30, 1536)),
            false,
        ),
        (
            "Qwen2-0.5B shape (plan only)",
            shape((151936, 2048, 896), (14, 2, 64), (24, 4864)),
            true,
        ),
        (
            "TinyLlama-1.1B shape (plan only)",
            shape((32000, 2048, 2048), (32, 4, 64), (22, 5632)),
            false,
        ),
    ];
    let limit = gpu.info().max_binding;
    for (name, c, biases) in refs {
        let inf = |q| plan(&c, q);
        let t = 128.min(c.ctx);
        let mut ft = static_plan_for(&param_sizes(&c, biases));
        for (n, b) in activation_plan(&c, 1, t).rows() {
            ft.add(n, *b);
        }
        let verdict = |p: &BufferPlan| match p.check(limit, None) {
            Ok(()) => "fits".to_string(),
            Err(e) => format!("refused: {e}"),
        };
        println!(
            "| {name} | {:.0}M | {} | {} | {} | {} | infer f32 {}; infer Q4_0 {}; fine-tune {} |",
            params(&c, biases) as f64 / 1e6,
            mib(inf(None).total()),
            mib(inf(Some(QKind::Q8_0)).total()),
            mib(inf(Some(QKind::Q4_0)).total()),
            mib(ft.total()),
            verdict(&inf(None)),
            verdict(&inf(Some(QKind::Q4_0))),
            verdict(&ft),
        );
    }
}
