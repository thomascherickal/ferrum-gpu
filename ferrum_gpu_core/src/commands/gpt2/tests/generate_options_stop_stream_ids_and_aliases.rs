use super::*;

#[test]
fn generate_options_stop_stream_ids_and_aliases() {
    let d = dir("gen");
    let (c, m) = (data(&d), d.join("m.gguf"));
    ok(&format!(
        "train {FAST} --epochs 40 --data {c} --out {} ",
        m.display()
    ));
    let g = |extra: &str| {
        ok(&format!(
            "generate --device cpu --model {} {extra}",
            m.display()
        ))
    };
    let base = g("--prompt the --max-new 30");
    // the same run with cpu's spellings gives the same text
    assert_eq!(base, g("--prompt the --max-tokens 30"));
    assert_eq!(base, g("--prompt the --chars 30"));
    assert_eq!(g("--prompt the --max-new 30 --temp 0 --gen-seed 5"), base);
    // a stop string ends the text before it
    let stopped = g("--prompt the --max-new 30 --stop he");
    assert!(
        stopped.len() < base.len() && !stopped["the".len()..].contains("he"),
        "{stopped}"
    );
    // streaming prints the same text, then a newline
    assert_eq!(g("--prompt the --max-new 30 --stream"), base);
    // sampling knobs run and are seeded
    let s1 = g("--prompt the --max-new 20 --temp 1 --top-p 0.9 --top-k 5 --repetition-penalty 1.2 --seed 3");
    assert_eq!(s1, g("--prompt the --max-new 20 --temp 1 --top-p 0.9 --top-k 5 --repetition-penalty 1.2 --seed 3"));
    // raw ids instead of a prompt
    let mut out = Vec::new();
    let by_ids = words(&format!(
        "generate --device cpu --model {} --max-new 3",
        m.display()
    ));
    let by_ids: Vec<String> = by_ids
        .into_iter()
        .chain(["--ids".into(), "100 101".into()])
        .collect();
    run(&by_ids, &mut out).unwrap();
    assert!(!out.is_empty());
    assert!(run(
        &words(&format!(
            "generate --device cpu --model {} --ids 1x",
            m.display()
        )),
        &mut Vec::new()
    )
    .unwrap_err()
    .contains("not an id"));
    assert!(err(&format!(
        "generate --device cpu --model {} --prompt {}",
        m.display(),
        "a".repeat(400)
    ))
    .contains("context"));
    let empty: Vec<String> = [
        "generate",
        "--device",
        "cpu",
        "--model",
        m.to_str().unwrap(),
        "--prompt",
        "",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    assert!(run(&empty, &mut Vec::new())
        .unwrap_err()
        .contains("prompt is empty"));
}
