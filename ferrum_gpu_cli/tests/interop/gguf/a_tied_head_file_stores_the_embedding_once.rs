use super::*;

#[test]
fn a_tied_head_file_stores_the_embedding_once() {
    // --tie writes no output.weight; llama.cpp then ties the head itself
    let f = fixture();
    let out = f.dir.join("tied.gguf");
    crate::support::model::ferrum_ok(&[
        "train",
        "--device",
        "cpu",
        "--context",
        "16",
        "--embed",
        "32",
        "--heads",
        "2",
        "--blocks",
        "1",
        "--hidden",
        "64",
        "--batch",
        "8",
        "--epochs",
        "2",
        "--vocab",
        "300",
        "--tie",
        "--quant",
        "f32",
        "--data",
        f.corpus.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    let (_, g) = load(&out);
    assert!(g.tensor("output.weight").is_none());
    assert!(g.tensor("token_embd.weight").is_some());
}
