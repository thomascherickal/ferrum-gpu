//! Memory pre-flight (6.4): list every device buffer a run will create, using the same plans the
//! GPU engine checks itself, and refuse before anything is opened or uploaded.

use ferrum_ref::llama::Llama;
use ferrum_wgpu::budget::{Budget, BufferPlan, Source};
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

/// The memory budget for `--gpu-mem-mb`; `None` means unknown, and then only the adapter's
/// binding limit is enforced.
// ponytail: only the override is used; VRAM from nvidia-smi/sysfs (`budget::resolve`) can be
// wired in when a run needs it.
pub fn budget(mem_mb: Option<u64>) -> Option<Budget> {
    mem_mb.map(|mb| Budget {
        bytes: mb * 1024 * 1024,
        source: Source::Override,
    })
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
        check(&p, &adapter(1 << 30), budget(Some(4096))).unwrap();
        assert!(report(&p, budget(Some(4096))).contains("budget 4096.0 MiB"));
        assert!(report(&p, None).contains("budget unknown"));
    }

    #[test]
    fn refuses_a_budget_that_is_too_small_and_says_what_to_do() {
        let m = fresh_model("tiny", 1).unwrap();
        let p = train_plan(&m, 2, 16);
        let e = check(&p, &adapter(1 << 30), budget(Some(0))).unwrap_err();
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
