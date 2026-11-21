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
