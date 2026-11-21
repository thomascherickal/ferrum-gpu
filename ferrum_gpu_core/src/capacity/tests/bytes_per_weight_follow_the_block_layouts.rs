use super::*;

#[test]
fn bytes_per_weight_follow_the_block_layouts() {
    assert_eq!(Precision::F32.bytes_per_weight(), 4.0);
    assert_eq!(Precision::Q8_0.bytes_per_weight(), 1.0625);
    assert_eq!(Precision::Q4_0.bytes_per_weight(), 0.5625);
    assert_eq!(Precision::ALL.map(Precision::name), ["Q4_0", "Q8_0", "f32"]);
}

#[test]
fn the_formulas_give_the_documented_examples() {
    // 16 GB discrete: 14.4 GB usable; Q4_0 holds 22.8B and a full f32 fine-tune 816M (hardware-caps.md)
    let r = row(16.0, false);
    assert!((r.usable - 14.4e9).abs() < 1.0);
    assert_eq!(format_params(r.infer[0]), "22.8B");
    assert_eq!(format_params(r.infer[1]), "12.1B");
    assert_eq!(format_params(r.infer[2]), "3.2B");
    assert_eq!(format_params(r.train), "816M");
    // the same memory as unified memory holds less
    let u = row(16.0, true);
    assert_eq!(format_params(u.infer[0]), "17.6B");
    assert!(u.usable < r.usable);
}

#[test]
fn what_fits_is_the_inverse_of_what_a_model_needs() {
    for p in Precision::ALL {
        let params = 7e9;
        let need = inference_bytes(params, p);
        assert!(
            (max_params_inference(need, p) - params).abs() < 1.0,
            "{p:?}"
        );
    }
    let need = training_bytes(1e9);
    assert!((max_params_training(need) - 1e9).abs() < 1.0);
    // a smaller precision always holds more, and training holds the least
    let r = row(24.0, false);
    assert!(r.infer[0] > r.infer[1] && r.infer[1] > r.infer[2] && r.infer[2] > r.train);
}

#[test]
fn memory_below_the_overhead_holds_nothing_and_never_goes_negative() {
    for gb in [0.0, 0.1, -5.0] {
        let r = row(gb, false);
        assert!(r.infer.iter().all(|v| *v == 0.0) && r.train == 0.0, "{gb}");
    }
    assert_eq!(max_params_inference(OVERHEAD_BYTES, Precision::Q4_0), 0.0);
    assert_eq!(format_params(0.0), "0M");
}

#[test]
fn the_decode_ceiling_is_bandwidth_over_weight_bytes() {
    // 936 GB/s over 7B weights at Q4_0 (3.94 GB): 237.7 tokens/s
    let t = decode_ceiling_tps(936.0, 7e9, Precision::Q4_0);
    assert!((t - 237.7).abs() < 0.1, "{t}");
    assert!(decode_ceiling_tps(936.0, 7e9, Precision::F32) < t / 7.0);
    assert_eq!(decode_ceiling_tps(936.0, 0.0, Precision::Q4_0), 0.0);
    assert_eq!(decode_ceiling_tps(0.0, 7e9, Precision::Q4_0), 0.0);
}

#[test]
fn params_are_written_in_millions_below_a_billion_and_billions_above() {
    assert_eq!(format_params(163e6), "163M");
    assert_eq!(format_params(999e6), "999M");
    assert_eq!(format_params(1e9), "1.0B");
    assert_eq!(format_params(1.78e9), "1.8B");
    assert_eq!(format_params(70e9), "70.0B");
}
