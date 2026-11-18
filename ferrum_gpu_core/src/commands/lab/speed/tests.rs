use super::*;

#[test]
fn sizes_parse_and_both_are_valid_models_with_whole_quantisation_blocks() {
    assert_eq!(Size::parse("small").unwrap(), Size::Small);
    assert_eq!(Size::parse("medium").unwrap(), Size::Medium);
    assert!(Size::parse("huge").unwrap_err().contains("small, medium"));
    for s in [Size::Small, Size::Medium] {
        let c = s.config();
        c.validate().unwrap();
        assert!(
            c.dim % 32 == 0 && c.hidden % 32 == 0 && c.q_dim() % 32 == 0,
            "{s:?}"
        );
        assert!(s.describe().contains(&format!("{} layers", c.layers)));
    }
    assert!(Size::Medium.config().layers > Size::Small.config().layers);
}

#[test]
fn tokens_per_second_from_milliseconds() {
    assert_eq!(tps(32, 1000.0), 32.0);
    assert_eq!(tps(32, 500.0), 64.0);
    assert_eq!(tps(32, 0.0), 0.0);
    assert_eq!(tps(32, -1.0), 0.0);
}

#[test]
fn the_token_ids_stay_inside_the_vocabulary() {
    for v in [17usize, 512, 2048] {
        let i = ids(v);
        assert_eq!(i.len(), TOKENS);
        assert!(i.iter().all(|&t| t < v));
    }
}

#[test]
fn the_cpu_reference_reports_a_speed() {
    let w = Llama::random(Size::Small.config(), 5, false, false);
    let v = cpu_prefill(&w, 1).unwrap();
    assert!(v.is_finite() && v > 0.0, "{v}");
}

#[test]
fn every_precision_measures_a_speed_on_an_available_adapter() {
    let Some(gpu) = ferrum_wgpu::device::open_for_test() else {
        return;
    };
    let (speeds, cpu) = run(&gpu, Size::Small, 1).unwrap();
    assert_eq!(
        speeds
            .iter()
            .map(|s| s.precision.as_str())
            .collect::<Vec<_>>(),
        ["f32", "F16", "Q8_0", "Q4_0", "MXFP4", "FP8 E4M3", "FP8 E5M2", "Q2_K"]
    );
    for s in &speeds {
        assert!(
            s.decode_tps > 0.0 && s.prefill_tps > 0.0 && s.model_mib > 0.0,
            "{s:?}"
        );
    }
    // f32 > F16 > Q8_0 > Q4_0 strictly; the formats added later are never larger than their
    // size class (FP8 is as big as Q8_0, MXFP4 as big as Q4_0, Q2_K smaller than Q4_0)
    let mib = |name: &str| {
        speeds
            .iter()
            .find(|s| s.precision == name)
            .unwrap()
            .model_mib
    };
    assert!(mib("f32") > mib("F16") && mib("F16") > mib("Q8_0") && mib("Q8_0") > mib("Q4_0"));
    assert!(
        (mib("FP8 E4M3") - mib("Q8_0")).abs() < 1e-9
            && (mib("FP8 E5M2") - mib("Q8_0")).abs() < 1e-9
    );
    assert!((mib("MXFP4") - mib("Q4_0")).abs() < 1e-9);
    assert!(mib("Q2_K") < mib("Q4_0"), "{speeds:?}");
    assert!(cpu > 0.0);
}
