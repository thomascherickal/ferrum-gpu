use super::*;

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
    let spy = FakeTool {
        log: Some(log.clone()),
        replies: vec![("ls".into(), listing)],
        ..FakeTool::default()
    };
    let lms = install(&d, "lms", spy);
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
    let broken = FakeTool {
        stderr: "broken".into(),
        exit_code: 1,
        ..FakeTool::default()
    };
    let fail = install(&d, "fail", broken);
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
    let blank = FakeTool {
        replies: vec![("ls".into(), "[]".into())],
        ..FakeTool::default()
    };
    let empty = install(&d, "empty", blank);
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
    // a tool that exists but cannot be started
    let noexec = d.join("noexec");
    std::fs::write(&noexec, "not a program").unwrap();
    let an = Args::parse(
        &[
            crate::commands::flags::val("--model"),
            crate::commands::flags::val("--bin"),
        ],
        &w(&format!("--model {m} --bin {}", noexec.display())),
    )
    .unwrap();
    assert!(export_lmstudio_with(&an, &mut Vec::new(), 1, std::time::Duration::ZERO).is_err());
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
