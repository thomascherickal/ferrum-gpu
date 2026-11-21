use super::*;

#[test]
fn prepare_writes_splits_and_a_manifest_and_explains_its_usage() {
    let d = dir("prepare");
    let c = d.join("c.txt");
    std::fs::write(
        &c,
        "one doc here<|endoftext|>two doc here<|endoftext|>two doc here",
    )
    .unwrap();
    let o = d.join("out");
    let s = ok(&format!(
        "prepare --input {} --out {} --dedup --nfkc --val-frac 0 --test-frac 0",
        c.display(),
        o.display()
    ));
    assert!(
        s.contains("train=2 val=0 test=0") && s.contains("duplicate=1"),
        "{s}"
    );
    let m = std::fs::read_to_string(o.join("manifest.json")).unwrap();
    assert!(m.contains("\"--dedup\"") && m.contains("\"sha256\""));
    assert!(err("prepare --out x").contains("usage: ferrum-gpu prepare"));
    assert!(err(&format!("prepare --input {}", c.display())).contains("usage: ferrum-gpu prepare"));
    assert!(err(&format!(
        "prepare --input {} --out {} --seed x",
        c.display(),
        o.display()
    ))
    .contains("--seed"));
}

#[test]
fn the_token_loop_stops_at_once_when_the_request_is_cancelled() {
    use ferrum_model::sampling::Params;
    use std::sync::atomic::{AtomicBool, Ordering};
    let flag = std::sync::Arc::new(AtomicBool::new(false));
    let _scope = crate::cancel::install(flag.clone());
    let o = gen::Opts {
        max_new: 50,
        params: Params {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        },
        seed: 1,
        stops: vec![],
        stream: false,
        stop_ids: vec![],
    };
    let mut fed = 0usize;
    let mut feed = |_: usize| -> Result<Vec<f32>, String> {
        fed += 1;
        if fed == 4 {
            flag.store(true, Ordering::Relaxed);
        }
        Ok(vec![0.0, 1.0, 0.5])
    };
    let e = gen::run(&mut feed, &|_| String::new(), 100, &[1, 2], &o, None).unwrap_err();
    assert_eq!(e, crate::cancel::MESSAGE);
    assert!(fed <= 5, "stopped within a token of the flag: fed {fed}");
}
