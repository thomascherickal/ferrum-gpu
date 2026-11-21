use super::*;

#[test]
fn the_core_depends_on_no_front_end() {
    let d = deps("ferrum_gpu_core");
    assert!(!d.is_empty(), "the core has engine dependencies");
    for f in FRONT_ENDS {
        assert!(
            !d.contains(&f.to_string()),
            "ferrum_gpu_core must not depend on {f}"
        );
    }
}

#[test]
fn the_cli_and_the_gui_each_depend_on_the_core_and_not_on_each_other() {
    for (krate, other) in [
        ("ferrum_gpu_cli", "ferrum_gpu_gui"),
        ("ferrum_gpu_gui", "ferrum_gpu_cli"),
    ] {
        let d = deps(krate);
        assert!(
            d.contains(&"ferrum_gpu_core".to_string()),
            "{krate} uses the core"
        );
        assert!(
            !d.contains(&other.to_string()),
            "{krate} must not depend on {other}"
        );
    }
}

#[test]
fn the_server_depends_on_the_core_only_among_the_front_ends() {
    let d = deps("ferrum_gpu_server");
    assert!(d.contains(&"ferrum_gpu_core".to_string()));
    for f in ["ferrum_gpu_gui", "ferrum_gpu_cli"] {
        assert!(
            !d.contains(&f.to_string()),
            "the server must not depend on {f}"
        );
    }
}

#[test]
fn the_desktop_app_is_its_own_workspace_outside_the_main_one_and_sits_on_the_core_and_the_gui() {
    let root_manifest = fs::read_to_string(root().join("Cargo.toml")).unwrap();
    assert!(
        root_manifest.contains("exclude") && root_manifest.contains("ferrum_gpu_studio"),
        "the main workspace excludes the app, so the Tauri tree stays out of its lockfile"
    );
    let studio = fs::read_to_string(root().join("ferrum_gpu_studio/Cargo.toml")).unwrap();
    assert!(
        studio.lines().any(|l| l.trim() == "[workspace]"),
        "the app declares its own workspace"
    );
    assert!(
        root().join("ferrum_gpu_studio/Cargo.lock").is_file(),
        "with its own lockfile"
    );
    let d = deps("ferrum_gpu_studio");
    for need in ["ferrum_gpu_core", "ferrum_gpu_gui"] {
        assert!(d.contains(&need.to_string()), "the app uses {need}");
    }
    assert!(
        !d.contains(&"ferrum_gpu_cli".to_string()),
        "the app runs the commands from the core, not the CLI"
    );
    // and nothing in the main workspace depends on the app
    for k in [
        "ferrum_gpu_core",
        "ferrum_gpu_cli",
        "ferrum_gpu_gui",
        "ferrum_gpu_server",
    ] {
        assert!(
            !deps(k).contains(&"ferrum_gpu_studio".to_string()),
            "{k} must not depend on the app"
        );
    }
}
