//! Model files that parse as GGUF but cannot be loaded, and a report that cannot be written.

use super::{err, scratch, trained};
use ferrum_gguf::{Gguf, GgufBuilder};

/// `bytes` rebuilt without the metadata keys named in `drop_keys` and without any tensor when
/// `drop_tensors`.
fn rebuilt(bytes: &[u8], drop_keys: &[&str], drop_tensors: bool) -> Vec<u8> {
    let g = Gguf::parse(bytes.to_vec()).unwrap();
    let mut b = GgufBuilder::new();
    for (k, v) in
        g.kv.iter()
            .filter(|(k, _)| !drop_keys.contains(&k.as_str()))
    {
        b.meta(k, v.clone());
    }
    for t in g.tensors.iter().filter(|_| !drop_tensors) {
        let data = g.tensor_bytes(&t.name).unwrap().to_vec();
        b.tensor_raw(&t.name, &t.dims, t.ggml_type, data);
    }
    b.into_bytes()
}

#[test]
fn a_gpt2_file_without_weights_or_with_a_broken_tokenizer_names_the_file() {
    let (d, c) = scratch("unusable_model");
    let (m, _) = trained(&d, &c, 300, "m");
    let bytes = std::fs::read(&m).unwrap();
    for (tag, file) in [
        ("noweights", rebuilt(&bytes, &[], true)),
        ("notok", rebuilt(&bytes, &["tokenizer.ggml.merges"], false)),
    ] {
        let p = d.join(format!("{tag}.gguf"));
        std::fs::write(&p, file).unwrap();
        let e = err(&format!(
            "generate --device cpu --model {} --prompt the",
            p.display()
        ));
        assert!(e.starts_with(&format!("{}: ", p.display())), "{tag}: {e}");
    }
}

#[test]
fn eval_names_a_report_file_it_cannot_write() {
    let (d, c) = scratch("eval_json");
    let (m, _) = trained(&d, &c, 300, "m");
    let json = "/nonexistent-ferrum-dir/e.json";
    let e = err(&format!(
        "eval --model {} --data {} --json {json}",
        m.display(),
        c.display()
    ));
    assert!(e.starts_with(&format!("{json}: ")), "{e}");
}
