use super::*;

#[test]
fn block_on_runs_ready_and_pending_futures() {
    assert_eq!(block_on(async { 7 }), 7);
    struct Twice(u8);
    impl Future for Twice {
        type Output = u8;
        fn poll(mut self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<u8> {
            self.0 += 1;
            if self.0 >= 3 {
                Poll::Ready(self.0)
            } else {
                Poll::Pending
            }
        }
    }
    assert_eq!(block_on(Twice(0)), 3);
}

#[test]
fn kind_mapping() {
    assert_eq!(kind_of(wgpu::DeviceType::DiscreteGpu), Kind::Discrete);
    assert_eq!(kind_of(wgpu::DeviceType::IntegratedGpu), Kind::Integrated);
    assert_eq!(kind_of(wgpu::DeviceType::Cpu), Kind::Software);
    assert_eq!(kind_of(wgpu::DeviceType::VirtualGpu), Kind::Other);
}

#[test]
fn listing_never_panics_and_unknown_name_errors() {
    let list = Gpu::list();
    eprintln!("adapters: {list:?}");
    let e = Gpu::open(Some("no-such-adapter-xyz"))
        .err()
        .expect("must fail");
    assert!(e.to_string().contains("no-such-adapter-xyz") || e.to_string().contains("no GPU"));
}

#[test]
fn open_reports_limits_when_a_gpu_exists() {
    let Some(g) = open_for_test() else { return };
    assert!(g.info().max_binding > 0);
    assert!(g.max_workgroups() >= 65535);
    assert!(g.info().max_workgroup_invocations >= 256);
}

#[test]
fn a_failed_stage_is_named_and_a_good_one_passes_through() {
    let e = stage(Err::<(), _>("device gone"), "poll failed").unwrap_err();
    assert_eq!(e.to_string(), "poll failed: device gone");
    assert_eq!(stage(Ok::<_, String>(3), "poll failed").unwrap(), 3);
}
