use super::*;

#[allow(clippy::module_inception)]
mod gate_tests {
    use super::*;
    use crate::engine::{Backend, CpuBackend, GenOpts, Progress, TrainSpec};
    use ferrum_wgpu::select::{AdapterSummary, Kind};

    /// Pretends to be a GPU named "Fake GPU"; never runs anything.
    struct FakeGpu;
    impl Backend for FakeGpu {
        fn kind(&self) -> &'static str {
            "gpu"
        }
        fn name(&self) -> String {
            "Fake GPU".into()
        }
        fn train(
            &self,
            _: &Llama,
            _: &TrainSpec,
            _: &Arc<Mutex<Progress>>,
        ) -> Result<Llama, String> {
            unreachable!()
        }
        fn generate(&self, _: &Llama, _: &[usize], _: &GenOpts) -> Result<Vec<usize>, String> {
            unreachable!()
        }
    }

    fn listed() -> Vec<AdapterSummary> {
        vec![AdapterSummary {
            name: "Fake GPU".into(),
            backend: "Vulkan".into(),
            kind: Kind::Discrete,
            max_buffer: 1 << 32,
            max_binding: 1 << 30,
            max_workgroup_invocations: 256,
            features: Default::default(),
        }]
    }

    fn app(b: Arc<dyn Backend>, adapters: Vec<AdapterSummary>) -> Session {
        Session::with_backend(b, DeviceChoice::Gpu, None, adapters)
    }

    #[test]
    fn a_gpu_run_that_fits_prints_the_estimate_and_one_that_does_not_is_refused() {
        let m = fresh_model("tiny", 1).unwrap();
        let plan = preflight::train_plan(&m, 2, 16);
        let gpu = app(Arc::new(FakeGpu), listed());
        let mut out = Vec::new();
        gate(&mut out, &gpu, &plan, Some(4096)).unwrap();
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("preflight: GPU buffers"));
        let e = gate(&mut Vec::new(), &gpu, &plan, Some(0)).unwrap_err();
        assert!(e.contains("refusing to start on Fake GPU"), "{e}");
    }

    /// A card with 2048 MiB total and 1989 MiB free, as `nvidia-smi` would say; no RAM figures.
    struct Mx330;
    impl sysmem::Readers for Mx330 {
        fn nvidia_smi(&self) -> Option<String> {
            Some("Fake GPU, 2048, 1989\n".into())
        }
        fn meminfo(&self) -> Option<String> {
            None
        }
        fn sysctl_memsize(&self) -> Option<String> {
            None
        }
        fn sysfs_vram_mb(&self) -> Vec<u64> {
            vec![]
        }
    }

    /// A plan of `mib` MiB in two buffers (each under the fake adapter's binding limit).
    fn plan_of(mib: u64) -> ferrum_wgpu::budget::BufferPlan {
        let mut p = ferrum_wgpu::budget::BufferPlan::default();
        p.add("first half", mib / 2 * 1048576);
        p.add("second half", (mib - mib / 2) * 1048576);
        p
    }

    #[test]
    fn the_preflight_uses_the_cards_free_memory_with_the_measured_headroom() {
        // capability.md: on this card plans up to 1449 MiB loaded and plans from 1721 MiB failed
        let gpu = app(Arc::new(FakeGpu), listed());
        for fits in [279u64, 956, 1449] {
            let mut out = Vec::new();
            gate_with(&Mx330, &mut out, &gpu, &plan_of(fits), None)
                .unwrap_or_else(|e| panic!("{fits} MiB should pass: {e}"));
            assert!(
                String::from_utf8(out).unwrap().contains("budget 1591"),
                "the budget is shown"
            );
        }
        for refused in [1721u64, 1925, 2454, 4286] {
            let e = gate_with(&Mx330, &mut Vec::new(), &gpu, &plan_of(refused), None)
                .expect_err(&format!("{refused} MiB should be refused"));
            assert!(e.contains("refusing to start on Fake GPU"), "{e}");
        }
        // the flag still overrides what the machine says
        gate_with(&Mx330, &mut Vec::new(), &gpu, &plan_of(1925), Some(4096)).unwrap();
    }

    #[test]
    fn with_no_memory_figure_only_the_binding_limit_applies_as_before() {
        struct Blind;
        impl sysmem::Readers for Blind {
            fn nvidia_smi(&self) -> Option<String> {
                None
            }
            fn meminfo(&self) -> Option<String> {
                None
            }
            fn sysctl_memsize(&self) -> Option<String> {
                None
            }
            fn sysfs_vram_mb(&self) -> Vec<u64> {
                vec![]
            }
        }
        let gpu = app(Arc::new(FakeGpu), listed());
        let mut out = Vec::new();
        // 4286 MiB in buffers that each respect the binding limit: only the missing budget is at issue
        let mut big = ferrum_wgpu::budget::BufferPlan::default();
        for i in 0..5 {
            big.add(&format!("part {i}"), 857 * 1048576);
        }
        gate_with(&Blind, &mut out, &gpu, &big, None).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("budget unknown"));
    }

    #[test]
    fn the_decoder_plan_is_split_at_the_chosen_adapters_binding_limit() {
        let m = fresh_model("tiny", 1).unwrap();
        let whole = preflight::decode_buffers(&m);
        let biggest = whole
            .rows()
            .iter()
            .filter(|r| r.0.contains("blk0.gate") || r.0 == "head" || r.0 == "tok_emb")
            .map(|r| r.1)
            .max()
            .unwrap();
        // a limit between the unsplittable buffers and the biggest weight
        let unsplittable = whole
            .rows()
            .iter()
            .filter(|r| r.0.contains("cache") || r.0.starts_with("scratch"))
            .map(|r| r.1)
            .max()
            .unwrap();
        let limit = (biggest / 2).max(unsplittable);
        assert!(limit < biggest, "the tiny preset must have a weight bigger than its own KV cache ({limit} vs {biggest})");
        let mut one = listed();
        one[0].max_binding = limit;
        let split = preflight::decode_buffers_for(&m, &app(Arc::new(FakeGpu), one));
        assert_eq!(split.total(), whole.total(), "splitting keeps the bytes");
        assert!(split.rows().len() > whole.rows().len());
        assert!(split.rows().iter().all(|r| r.1 <= limit));
        // a CPU backend has no limit, so the plan is the whole one
        let cpu = preflight::decode_buffers_for(&m, &app(Arc::new(CpuBackend::new(1)), listed()));
        assert_eq!(cpu.rows(), whole.rows());
    }

    #[test]
    fn cpu_backends_and_unlisted_adapters_pass_silently() {
        let m = fresh_model("tiny", 1).unwrap();
        let plan = preflight::train_plan(&m, 2, 16);
        let cpu = app(Arc::new(CpuBackend::new(1)), listed());
        let mut out = Vec::new();
        gate(&mut out, &cpu, &plan, Some(0)).unwrap();
        gate(&mut out, &app(Arc::new(FakeGpu), vec![]), &plan, Some(0)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn mem_flag_parses_or_names_itself_in_the_error() {
        let ok = Args::parse(&DEVICE, &a("--gpu-mem-mb 512")).unwrap();
        assert_eq!(mem_mb(&ok).unwrap(), Some(512));
        assert_eq!(mem_mb(&Args::parse(&DEVICE, &[]).unwrap()).unwrap(), None);
        let bad = Args::parse(&DEVICE, &a("--gpu-mem-mb lots")).unwrap();
        assert!(mem_mb(&bad).unwrap_err().contains("--gpu-mem-mb"));
    }
}
