use super::*;

#[test]
fn tools_are_found_in_flag_env_path_then_home_order() {
    let d = dir("find");
    let t = d.join("tool");
    std::fs::write(&t, "x").unwrap();
    let tool = t.to_str().unwrap();
    let path_var: OsString = d.clone().into_os_string();
    assert_eq!(
        find_tool_in(Some(tool), None, "E", None, "tool", "x").unwrap(),
        t
    );
    assert!(find_tool_in(Some("/no/such"), None, "E", None, "tool", "x")
        .unwrap_err()
        .contains("--bin /no/such"));
    assert_eq!(
        find_tool_in(None, Some(tool), "E", None, "tool", "x").unwrap(),
        t
    );
    assert!(
        find_tool_in(None, Some("/no/such"), "FERRUM_X", None, "tool", "x")
            .unwrap_err()
            .contains("FERRUM_X=/no/such")
    );
    assert_eq!(
        find_tool_in(None, Some(""), "E", Some(&path_var), "tool", "x").unwrap(),
        t,
        "an empty variable is ignored"
    );
    let e = find_tool_in(
        None,
        None,
        "FERRUM_X",
        Some(&OsString::from("/nonexistent")),
        "tool",
        "no/home/fallback",
    )
    .unwrap_err();
    assert!(
        e.contains("cannot find `tool`") && e.contains("FERRUM_X"),
        "{e}"
    );
}

#[test]
fn the_lms_listing_is_matched_by_path_suffix() {
    let json = "Waking up LM Studio service...\n[{\"path\":\"other/x.gguf\",\"modelKey\":\"no\"},{\"path\":\"/home/u/.lmstudio/models/ferrum/m/m.gguf\",\"modelKey\":\"ferrum/m\"}]";
    assert_eq!(
        model_key_for(json, "ferrum/m/m.gguf").as_deref(),
        Some("ferrum/m")
    );
    assert_eq!(
        model_key_for(json, "errum/m/m.gguf"),
        None,
        "the match is on a / boundary"
    );
    assert_eq!(
        model_key_for(
            "{\"path\":\"ferrum/m/m.gguf\",\"modelKey\":\"k\"}",
            "ferrum/m/m.gguf"
        )
        .as_deref(),
        Some("k")
    );
    assert_eq!(model_key_for("no json at all", "x"), None);
    assert_eq!(model_key_for("[]", "x"), None);
    assert_eq!(
        model_key_for("[{\"path\":\"a/b\"}]", "a/b"),
        None,
        "an entry without a modelKey"
    );
}

#[cfg(unix)]
#[test]
fn export_lmstudio_imports_then_waits_for_the_listing() {
    let d = dir("lms");
    let m = model(&d);
    let name = Path::new(&m)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let log = d.join("log.txt");
    let listing = format!("[{{\"path\":\"/x/ferrum/m/{name}\",\"modelKey\":\"ferrum/m\"}}]");
    let lms = script(
        &d.join("lms"),
        &format!(
            "echo \"$@\" >> {}\nif [ \"$1\" = ls ]; then echo '{listing}'; fi",
            log.display()
        ),
    );
    let a = Args::parse(
        &[
            crate::commands::flags::val("--model"),
            crate::commands::flags::val("--repo"),
            crate::commands::flags::val("--bin"),
        ],
        &w(&format!("--model {m} --repo ferrum/m --bin {lms}")),
    )
    .unwrap();
    let mut o = Vec::new();
    export_lmstudio_with(&a, &mut o, 3, std::time::Duration::from_millis(1)).unwrap();
    assert!(String::from_utf8(o).unwrap().contains("'ferrum/m'"));
    let calls = std::fs::read_to_string(&log).unwrap();
    assert!(
        calls.starts_with("import -y -l --user-repo ferrum/m ") && calls.contains("\nls --json"),
        "{calls}"
    );
    // an import that fails, and a listing that never shows the model
    let fail = script(&d.join("fail"), "echo broken >&2; exit 1");
    let af = Args::parse(
        &[
            crate::commands::flags::val("--model"),
            crate::commands::flags::val("--bin"),
        ],
        &w(&format!("--model {m} --bin {fail}")),
    )
    .unwrap();
    assert!(
        export_lmstudio_with(&af, &mut Vec::new(), 1, std::time::Duration::ZERO)
            .unwrap_err()
            .contains("`lms import` failed")
    );
    let empty = script(&d.join("empty"), "if [ \"$1\" = ls ]; then echo '[]'; fi");
    let ae = Args::parse(
        &[
            crate::commands::flags::val("--model"),
            crate::commands::flags::val("--bin"),
        ],
        &w(&format!("--model {m} --bin {empty}")),
    )
    .unwrap();
    assert!(
        export_lmstudio_with(&ae, &mut Vec::new(), 2, std::time::Duration::from_millis(1))
            .unwrap_err()
            .contains("never listed it")
    );
    // without --repo the repo is ferrum/<file stem>
    let ad = Args::parse(
        &[
            crate::commands::flags::val("--model"),
            crate::commands::flags::val("--bin"),
        ],
        &w(&format!("--model {m} --bin {lms}")),
    )
    .unwrap();
    export_lmstudio_with(&ad, &mut Vec::new(), 3, std::time::Duration::from_millis(1)).unwrap();
    assert!(std::fs::read_to_string(&log)
        .unwrap()
        .contains("--user-repo ferrum/m "));
}
