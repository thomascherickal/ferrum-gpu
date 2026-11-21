use super::*;

#[test]
fn the_registry_lists_exactly_the_shader_files_on_disk() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let mut on_disk = Vec::new();
    for group in std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .filter(|g| g.path().is_dir())
    {
        for f in std::fs::read_dir(group.path()).unwrap().flatten() {
            let name = f.file_name().to_string_lossy().into_owned();
            if let Some(stem) = name.strip_suffix(".wgsl") {
                on_disk.push(format!("{}/{stem}", group.file_name().to_string_lossy()));
            }
        }
    }
    on_disk.sort();
    let registered: Vec<String> = SOURCES.iter().map(|(n, _)| n.to_string()).collect();
    assert_eq!(
        registered, on_disk,
        "registry and src/shaders/ differ: regenerate shaders.rs"
    );
    let mut dedup = registered.clone();
    dedup.dedup();
    assert_eq!(dedup.len(), registered.len(), "names are unique");
}

#[test]
fn every_registered_kernel_parses_and_validates_as_wgsl_with_the_prelude() {
    for (name, src) in SOURCES {
        let code = format!("{PRELUDE}\n{src}");
        let module = naga::front::wgsl::parse_str(&code)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
        let flags = naga::valid::ValidationFlags::all();
        naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
}

#[test]
fn lookup_finds_registered_kernels_and_refuses_others() {
    assert!(source("llama/rmsnorm").unwrap().contains("fn main"));
    assert!(source("llama/../rmsnorm").is_none());
    assert!(source("").is_none());
    assert!(PRELUDE.contains("fn gid"));
}
