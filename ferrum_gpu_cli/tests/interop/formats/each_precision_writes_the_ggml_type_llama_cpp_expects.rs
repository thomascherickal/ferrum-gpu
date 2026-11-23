use super::*;

#[test]
fn each_precision_writes_the_ggml_type_llama_cpp_expects() {
    let w = wide();
    for (label, path, want) in [
        ("f16", &w.f16, F16),
        ("fp4", &w.fp4, MXFP4),
        ("q2", &w.q2, Q2_K),
    ] {
        let (mats, vecs) = types(&load(path));
        assert!(!mats.is_empty(), "{label}");
        // every matrix of this model is a whole number of blocks, so none falls back
        assert!(mats.iter().all(|&t| t == want), "{label}: {mats:?}");
        assert!(
            vecs.iter().all(|&t| t == 0),
            "{label}: vectors stay f32: {vecs:?}"
        );
    }
}

#[test]
fn llama_cpp_continues_the_prompt_exactly_as_we_do_for_f16_fp4_and_q2() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    for (label, path) in wide().all() {
        let (theirs, mine) = (completion(&dir, path), ours(path));
        assert_eq!(
            theirs, mine,
            "{label}: the two readers disagree about one file"
        );
        assert!(mine.starts_with(PROMPT), "{label}: {mine:?}");
    }
}

#[test]
fn our_q2_k_is_as_accurate_as_llama_quantizes() {
    let Some(dir) = llamacpp_dir() else {
        skip!("llama.cpp not installed (set FERRUM_LLAMACPP_DIR)")
    };
    let w = wide();
    let theirs = w.dir.join("lq-q2_k.gguf");
    let o = with_deadline(dir.join("llama-quantize"), 120)
        .env("LD_LIBRARY_PATH", &dir)
        .arg(&w.f32)
        .arg(&theirs)
        .arg("Q2_K")
        .stdin(Stdio::null())
        .output()
        .expect("run llama-quantize");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let (reference, mine, lq) = (load(&w.f32), load(&w.q2), load(&theirs));
    let mut compared = 0;
    for t in &mine.tensors {
        // only the tensors both of us wrote as Q2_K are comparable
        if t.ggml_type != Q2_K || lq.tensor(&t.name).map(|x| x.ggml_type) != Some(Q2_K) {
            continue;
        }
        let exact = reference.tensor_f32(&t.name).unwrap();
        let (a, b) = (
            mine.tensor_f32(&t.name).unwrap(),
            lq.tensor_f32(&t.name).unwrap(),
        );
        let (ours, theirs) = (rmse(&exact, &a), rmse(&exact, &b));
        // the same algorithm: within a few percent either way (float contraction may differ)
        assert!(
            ours <= theirs * 1.05 && theirs <= ours * 1.05,
            "{}: ours {ours} theirs {theirs}",
            t.name
        );
        let (x, y) = (
            mine.tensor_bytes(&t.name).unwrap(),
            lq.tensor_bytes(&t.name).unwrap(),
        );
        let same = x.iter().zip(y).filter(|(p, q)| p == q).count();
        crate::support::log::step(format!(
            "Q2_K {}: {} of {} bytes equal llama-quantize's, rmse ours {ours:.6} theirs {theirs:.6}",
            t.name,
            same,
            x.len()
        ));
        compared += 1;
    }
    assert!(compared > 0, "no tensor was Q2_K in both files");
}
