use super::fixtures::*;
use super::*;
use crate::engine::CpuBackend;

#[test]
fn cpu_backends_and_unlisted_adapters_pass_silently() {
    let m = fresh_model("tiny", 1).unwrap();
    let plan = preflight::train_plan(&m, 2, 16);
    let cpu = session_on(Arc::new(CpuBackend::new(1)), listed());
    let mut out = Vec::new();
    gate(&mut out, &cpu, &plan, Some(0)).unwrap();
    gate(
        &mut out,
        &session_on(Arc::new(FakeGpu), vec![]),
        &plan,
        Some(0),
    )
    .unwrap();
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
