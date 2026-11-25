use super::*;

#[test]
fn the_registry_lists_exactly_the_shader_files_on_disk() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let mut on_disk = Vec::new();
    let mut fragments = Vec::new();
    for group in std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .filter(|g| g.path().is_dir())
    {
        let group_name = group.file_name().to_string_lossy().into_owned();
        let list = if group_name == "common" {
            &mut fragments
        } else {
            &mut on_disk
        };
        for f in std::fs::read_dir(group.path()).unwrap().flatten() {
            let name = f.file_name().to_string_lossy().into_owned();
            if let Some(stem) = name.strip_suffix(".wgsl") {
                list.push(format!("{group_name}/{stem}"));
            }
        }
    }
    on_disk.sort();
    fragments.sort();
    let shared: Vec<String> = FRAGMENTS.iter().map(|(n, _)| n.to_string()).collect();
    assert_eq!(
        shared, fragments,
        "fragment list and src/shaders/common/ differ"
    );
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
fn every_composed_kernel_parses_and_validates_as_wgsl_with_the_prelude() {
    for (name, _) in SOURCES {
        let code = format!("{PRELUDE}\n{}", kernel(name).unwrap());
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
    assert!(kernel("nope/nope").unwrap_err().contains("no WGSL kernel"));
    assert!(kernel("matmul/matmul_blocked4t")
        .unwrap()
        .contains("fn load_a"));
    assert!(PRELUDE.contains("fn gid"));
}
