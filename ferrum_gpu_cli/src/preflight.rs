//! Memory pre-flight (6.4): list every device buffer a run will create, using the same plans the
//! GPU engine checks itself, and refuse before anything is opened or uploaded.

use ferrum_ref::llama::Llama;
use ferrum_wgpu::budget::{Budget, BufferPlan};
use ferrum_wgpu::llama::plan as decode_plan;
use ferrum_wgpu::llama_train::{activation_plan, static_plan};
use ferrum_wgpu::select::AdapterSummary;

/// Buffers a training run needs: weights, gradients, both Adam moments and the activations of a
/// `(batch, context)` step.
pub fn train_plan(m: &Llama, batch: usize, context: usize) -> BufferPlan {
    let mut p = static_plan(m);
    for (n, b) in activation_plan(&m.cfg, batch, context).rows() {
        p.add(n, *b);
    }
    p
}

/// Buffers a decoder needs: weights, tables, KV cache and scratch.
pub fn decode_buffers(m: &Llama) -> BufferPlan {
    decode_plan(&m.cfg, None)
}

/// Buffers a GPT-2 decoder needs: weights, KV cache and scratch.
pub fn gpt2_decode_buffers(m: &ferrum_ref::gpt2::Gpt2) -> BufferPlan {
    ferrum_wgpu::gpt2::plan(&m.cfg)
}

/// The memory budget: `--gpu-mem-mb`, else the VRAM `nvidia-smi` reports (`tool_mb`); `None` means
/// unknown, and then only the adapter's binding limit is enforced.
// ponytail: NVIDIA only; sysfs (AMD) and half-of-RAM for integrated adapters exist in
// `budget::resolve` and can be wired in when a run needs them.
pub fn budget(mem_mb: Option<u64>, tool_mb: Option<u64>) -> Option<Budget> {
    ferrum_wgpu::budget::resolve(mem_mb, tool_mb, false, None)
}

/// The total VRAM in MiB that `nvidia-smi` (csv, no header, no units: `name, total, used`) lists
/// for the adapter called `adapter`.
pub fn vram_from_smi(smi: &str, adapter: &str) -> Option<u64> {
    ferrum_wgpu::budget::parse_nvidia_smi(smi)
        .into_iter()
        .find(|(name, _)| adapter.contains(name.as_str()) || name.contains(adapter))
        .and_then(|(_, mb)| mb)
}

/// Ask `nvidia-smi` for the adapter's VRAM; `None` when the tool is absent or does not list it.
pub fn nvidia_vram_mb(adapter: &str) -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    vram_from_smi(&String::from_utf8_lossy(&out.stdout), adapter)
}

/// Mebibytes, one decimal.
fn mib(b: u64) -> String {
    format!("{:.1} MiB", b as f64 / 1048576.0)
}

/// The line to print on success.
pub fn report(plan: &BufferPlan, budget: Option<Budget>) -> String {
    let have = budget.map_or("unknown".to_string(), |b| mib(b.bytes));
    format!(
        "preflight: GPU buffers {} in {} allocations (budget {have})",
        mib(plan.total()),
        plan.rows().len()
    )
}

/// Refuse if `plan` does not fit `adapter` and `budget`. The message says what to do instead.
pub fn check(
    plan: &BufferPlan,
    adapter: &AdapterSummary,
    budget: Option<Budget>,
) -> Result<(), String> {
    plan.check(adapter.max_binding, budget).map_err(|e| {
        format!(
            "refusing to start on {}: {e} (needs {} in total); use a smaller preset, batch or \
             context, raise --gpu-mem-mb, or run with --device cpu",
            adapter.name,
            mib(plan.total())
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_gpu_gui::engine::fresh_model;
    use ferrum_wgpu::select::Kind;

    #[test]
    fn the_budget_is_the_override_then_the_tool_then_unknown() {
        use ferrum_wgpu::budget::Source;
        let b = budget(Some(100), Some(2048)).unwrap();
        assert_eq!((b.bytes, b.source), (100 << 20, Source::Override));
        let b = budget(None, Some(2048)).unwrap();
        assert_eq!((b.bytes, b.source), (2048 << 20, Source::Tool));
        assert!(budget(None, None).is_none());
    }

    #[test]
    fn nvidia_smi_vram_is_matched_to_the_adapter_by_name() {
        let smi = "NVIDIA GeForce MX330, 2048, 11\nNVIDIA A100, 40960, 0\n";
        assert_eq!(vram_from_smi(smi, "NVIDIA GeForce MX330"), Some(2048));
        assert_eq!(vram_from_smi(smi, "NVIDIA A100"), Some(40960));
        assert_eq!(vram_from_smi(smi, "AMD Radeon"), None);
        assert_eq!(vram_from_smi("NVIDIA X, [N/A], 0", "NVIDIA X"), None);
        assert_eq!(vram_from_smi("", "NVIDIA X"), None);
        // the real tool, if present, answers or declines without failing
        let _ = nvidia_vram_mb("NVIDIA GeForce MX330");
    }

    #[test]
    fn gpt2_decode_plan_lists_weights_and_cache() {
        let m = ferrum_ref::gpt2::Gpt2::init(
            ferrum_ref::gpt2::Config {
                vocab: 20,
                ctx: 8,
                dim: 8,
                heads: 2,
                layers: 1,
                hidden: 16,
            },
            1,
        );
        let p = gpt2_decode_buffers(&m);
        assert!(p.total() > 0 && !p.rows().is_empty());
    }

    fn adapter(max_binding: u64) -> AdapterSummary {
        AdapterSummary {
            name: "Fake GPU".into(),
            backend: "Vulkan".into(),
            kind: Kind::Discrete,
            max_buffer: max_binding * 4,
            max_binding,
            max_workgroup_invocations: 256,
        }
    }

    #[test]
    fn training_needs_more_than_decoding_and_grows_with_the_batch() {
        let m = fresh_model("tiny", 1).unwrap();
        let (t1, t8, d) = (
            train_plan(&m, 1, 16),
            train_plan(&m, 8, 16),
            decode_buffers(&m),
        );
        assert!(t8.total() > t1.total() && t1.total() > d.total() / 2);
        assert!(d.total() > 0);
    }

    #[test]
    fn fits_without_a_budget_and_with_a_generous_one() {
        let m = fresh_model("tiny", 1).unwrap();
        let p = train_plan(&m, 2, 16);
        check(&p, &adapter(1 << 30), None).unwrap();
        check(&p, &adapter(1 << 30), budget(Some(4096), None)).unwrap();
        assert!(report(&p, budget(Some(4096), None)).contains("budget 4096.0 MiB"));
        assert!(report(&p, None).contains("budget unknown"));
    }

    #[test]
    fn refuses_a_budget_that_is_too_small_and_says_what_to_do() {
        let m = fresh_model("tiny", 1).unwrap();
        let p = train_plan(&m, 2, 16);
        let e = check(&p, &adapter(1 << 30), budget(Some(0), None)).unwrap_err();
        assert!(
            e.contains("Fake GPU")
                && e.contains("GPU memory")
                && e.contains("--gpu-mem-mb")
                && e.contains("--device cpu"),
            "{e}"
        );
    }

    #[test]
    fn refuses_a_buffer_over_the_binding_limit() {
        let m = fresh_model("tiny", 1).unwrap();
        let e = check(&train_plan(&m, 2, 16), &adapter(1024), None).unwrap_err();
        assert!(e.contains("binding limit"), "{e}");
    }
}
