use super::*;
use ferrum_wgpu::device::select::Kind;

fn ad(name: &str, kind: Kind) -> AdapterSummary {
    AdapterSummary {
        name: name.into(),
        backend: "Vulkan".into(),
        kind,
        max_buffer: 1 << 30,
        max_binding: 1 << 27,
        max_workgroup_invocations: 256,
        features: Default::default(),
    }
}

fn table() -> Vec<AdapterSummary> {
    vec![
        ad("llvmpipe", Kind::Software),
        ad("Xe", Kind::Integrated),
        ad("MX330", Kind::Discrete),
    ]
}

#[test]
fn cpu_is_always_cpu() {
    for w in [Workload::Train, Workload::Decode] {
        let d = decide(DeviceChoice::Cpu, w, &table(), None).unwrap();
        assert_eq!(
            (d.choice, d.adapter, d.note),
            (DeviceChoice::Cpu, None, None)
        );
    }
}

#[test]
fn explicit_gpu_picks_the_best_or_the_named_adapter_or_fails() {
    let d = decide(DeviceChoice::Gpu, Workload::Decode, &table(), None).unwrap();
    assert_eq!(
        (d.choice, d.adapter.as_deref()),
        (DeviceChoice::Gpu, Some("MX330"))
    );
    let d = decide(DeviceChoice::Gpu, Workload::Train, &table(), Some("llvm")).unwrap();
    assert_eq!(d.adapter.as_deref(), Some("llvmpipe"));
    assert!(decide(DeviceChoice::Gpu, Workload::Train, &[], None)
        .unwrap_err()
        .contains("no GPU"));
    assert!(
        decide(DeviceChoice::Gpu, Workload::Train, &table(), Some("zzz"))
            .unwrap_err()
            .contains("no adapter matches")
    );
}

#[test]
fn auto_trains_on_hardware_and_decodes_on_the_cpu() {
    let d = decide(DeviceChoice::Auto, Workload::Train, &table(), None).unwrap();
    assert_eq!(
        (d.choice, d.adapter.as_deref(), d.note),
        (DeviceChoice::Auto, Some("MX330"), None)
    );
    let d = decide(DeviceChoice::Auto, Workload::Decode, &table(), None).unwrap();
    assert_eq!(d.choice, DeviceChoice::Cpu);
    assert!(d.note.unwrap().contains("34M"));
}

#[test]
fn auto_falls_back_to_the_cpu_with_one_line_when_there_is_no_hardware() {
    for (list, why) in [
        (vec![], "no GPU adapters"),
        (vec![ad("llvmpipe", Kind::Software)], "only software"),
    ] {
        let d = decide(DeviceChoice::Auto, Workload::Train, &list, None).unwrap();
        assert_eq!((d.choice, d.adapter), (DeviceChoice::Cpu, None));
        let n = d.note.unwrap();
        assert!(
            n.contains(why) && n.ends_with("using the CPU") && !n.contains('\n'),
            "{n}"
        );
    }
}
