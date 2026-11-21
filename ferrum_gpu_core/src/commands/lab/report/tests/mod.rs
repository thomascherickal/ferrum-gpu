use super::rate_m::*;
use super::*;

fn adapter() -> AdapterFacts {
    AdapterFacts {
        name: "Fake GPU".into(),
        backend: "Metal".into(),
        kind: "Integrated".into(),
        driver: "fake-driver 1.2".into(),
        max_binding_mib: 2047,
        features: vec!["f16".into(), "subgroups".into()],
        budget: "memory budget: 7000 MiB (an estimate)".into(),
    }
}

fn report() -> Report {
    Report {
        when: "2026-11-17".into(),
        os: "linux x86_64".into(),
        wgpu: "30.0.1".into(),
        adapter: Some(adapter()),
        model: "random 4-layer model".into(),
        reps: 5,
        checks: vec![
            Check {
                name: "f32".into(),
                error: 2.0e-7,
                tolerance: 1e-5,
            },
            Check {
                name: "q8_0".into(),
                error: 4.0e-7,
                tolerance: 1e-5,
            },
        ],
        speeds: vec![
            Speed {
                precision: "f32".into(),
                model_mib: 12.5,
                decode_tps: 123.4,
                prefill_tps: 4321.0,
            },
            Speed {
                precision: "q4_0".into(),
                model_mib: 3.1,
                decode_tps: 8.55,
                prefill_tps: 45.6,
            },
        ],
        cpu_prefill_tps: Some(900.0),
        notes: vec!["random weights; speed only".into()],
    }
}

mod a_check_passes_only_below_its_tolerance_and_never_with_nan;
