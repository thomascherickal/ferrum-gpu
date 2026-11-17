use super::*;
use crate::device::features_of;

#[test]
fn an_adapter_with_no_features_lists_none_and_the_default_is_that() {
    let none = AdapterFeatures::default();
    assert!(none.words().is_empty());
    assert_eq!(features_of(wgpu::Features::empty()), none);
}

#[test]
fn each_wgpu_feature_maps_to_its_own_flag_and_word() {
    for (f, word) in [
        (wgpu::Features::SHADER_F16, "f16"),
        (wgpu::Features::SUBGROUP, "subgroups"),
        (
            wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
            "cooperative-matrix",
        ),
        (wgpu::Features::TIMESTAMP_QUERY, "timestamps"),
    ] {
        let got = features_of(f);
        assert_eq!(got.words(), vec![word], "{word}");
    }
}

#[test]
fn several_features_are_listed_in_a_fixed_order() {
    let f = features_of(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP,
    );
    assert_eq!(f.words(), vec!["f16", "subgroups", "timestamps"]);
    assert!(f.shader_f16 && f.subgroups && f.timestamp_query && !f.cooperative_matrix);
}
