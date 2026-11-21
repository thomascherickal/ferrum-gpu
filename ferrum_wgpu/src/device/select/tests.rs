use super::*;

fn a(name: &str, kind: Kind) -> AdapterSummary {
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

#[test]
fn never_picks_an_opengl_adapter_unprompted() {
    let mut gl = a("Mesa Intel", Kind::Integrated);
    gl.backend = "Gl".into();
    assert_eq!(choose(&[gl.clone()], None), Err(SelectError::OnlySoftware));
    let vk = a("Intel Xe", Kind::Integrated);
    assert_eq!(choose(&[gl.clone(), vk], None), Ok(1));
    assert_eq!(choose(&[gl], Some("mesa")), Ok(0));
}

#[test]
fn prefers_discrete_over_integrated() {
    let l = [
        a("Intel UHD", Kind::Integrated),
        a("NVIDIA MX330", Kind::Discrete),
    ];
    assert_eq!(choose(&l, None), Ok(1));
}

#[test]
fn falls_back_to_integrated_then_other() {
    assert_eq!(
        choose(
            &[
                a("llvmpipe", Kind::Software),
                a("Apple M2", Kind::Integrated)
            ],
            None
        ),
        Ok(1)
    );
    assert_eq!(choose(&[a("x", Kind::Other)], None), Ok(0));
}

#[test]
fn never_picks_software_unprompted() {
    assert_eq!(
        choose(&[a("llvmpipe", Kind::Software)], None),
        Err(SelectError::OnlySoftware)
    );
}

#[test]
fn a_name_picks_software_and_is_case_insensitive() {
    let l = [
        a("NVIDIA MX330", Kind::Discrete),
        a("llvmpipe (LLVM 17)", Kind::Software),
    ];
    assert_eq!(choose(&l, Some("LLVMPIPE")), Ok(1));
}

#[test]
fn unknown_name_lists_what_exists() {
    let l = [a("NVIDIA MX330", Kind::Discrete)];
    let e = choose(&l, Some("radeon")).unwrap_err();
    assert!(e.to_string().contains("NVIDIA MX330") && e.to_string().contains("radeon"));
}

#[test]
fn empty_list_is_an_error() {
    let e = choose(&[], None).unwrap_err();
    assert_eq!(e, SelectError::NoAdapters);
    assert!(e.to_string().contains("no GPU"));
    assert!(choose(&[], Some("x")).is_err());
    assert!(SelectError::OnlySoftware.to_string().contains("software"));
}
