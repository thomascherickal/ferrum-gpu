//! The GGUF files we write, checked against what llama.cpp's loader requires. No tool is run: this
//! is the container (header, alignment, offsets), the metadata keys and the tensor names, shapes and
//! types of a `gpt2` model, so a regression shows up here before it shows up as a failed load.

use crate::support::model::fixture;
use ferrum_gguf::Gguf;

const F32: u32 = 0;
const Q4_0: u32 = 2;
const Q8_0: u32 = 8;

fn load(p: &std::path::Path) -> (Vec<u8>, Gguf) {
    let bytes = std::fs::read(p).unwrap();
    let g = Gguf::parse(bytes.clone()).unwrap();
    (bytes, g)
}

#[test]
fn the_header_is_gguf_version_3() {
    for (label, p) in fixture().all() {
        let (b, _) = load(p);
        assert_eq!(&b[..4], b"GGUF", "{label}");
        assert_eq!(
            u32::from_le_bytes(b[4..8].try_into().unwrap()),
            3,
            "{label}: version"
        );
    }
}

#[test]
fn tensor_data_is_aligned_ordered_and_inside_the_file() {
    for (label, p) in fixture().all() {
        let (b, g) = load(p);
        let align = 32u64;
        assert_eq!(g.data_offset() as u64 % align, 0, "{label}: data start");
        let mut end = 0u64;
        let mut tensors: Vec<_> = g.tensors.iter().collect();
        tensors.sort_by_key(|t| t.offset);
        for t in tensors {
            assert_eq!(t.offset % align, 0, "{label}: {} offset", t.name);
            assert!(
                t.offset >= end,
                "{label}: {} overlaps the one before",
                t.name
            );
            let bytes = g.tensor_bytes(&t.name).unwrap().len() as u64;
            end = t.offset + bytes;
            assert!(
                g.data_offset() as u64 + end <= b.len() as u64,
                "{label}: {} past the end",
                t.name
            );
        }
    }
}

#[test]
fn the_metadata_llama_cpp_reads_for_gpt2_is_present_and_consistent() {
    let (_, g) = load(&fixture().f32);
    assert_eq!(g.string("general.architecture").unwrap(), "gpt2");
    assert!(g.string("general.name").is_ok());
    let (ctx, dim, blocks) = (
        g.uint("gpt2.context_length").unwrap(),
        g.uint("gpt2.embedding_length").unwrap(),
        g.uint("gpt2.block_count").unwrap(),
    );
    assert_eq!((ctx, dim, blocks), (16, 32, 2));
    assert_eq!(g.uint("gpt2.feed_forward_length").unwrap(), 64);
    assert_eq!(g.uint("gpt2.attention.head_count").unwrap(), 2);
    assert!(g.get("gpt2.attention.layer_norm_epsilon").is_some());
}

#[test]
fn the_tokenizer_is_embedded_the_way_llama_cpp_reads_a_gpt2_vocabulary() {
    for (label, p) in fixture().all() {
        let (_, g) = load(p);
        assert_eq!(g.string("tokenizer.ggml.model").unwrap(), "gpt2", "{label}");
        let vocab = match g.get("tokenizer.ggml.tokens") {
            Some(ferrum_gguf::Value::Arr(a)) => a.len(),
            other => panic!("{label}: tokens {other:?}"),
        };
        assert_eq!(
            g.tensor("token_embd.weight").unwrap().dims,
            [32, vocab as u64],
            "{label}: [dim, vocab]"
        );
        assert!(g.get("tokenizer.ggml.merges").is_some(), "{label}: merges");
        assert!(
            g.uint("tokenizer.ggml.eos_token_id").is_ok(),
            "{label}: eos"
        );
    }
}

#[test]
fn every_tensor_llama_cpp_expects_is_there_with_ggml_dimension_order() {
    let (_, g) = load(&fixture().f32);
    let vocab = g.tensor("token_embd.weight").unwrap().dims[1];
    // ggml order: dimension 0 (contiguous) is the input width
    let expect: Vec<(String, Vec<u64>)> = {
        let mut v = vec![
            ("token_embd.weight".into(), vec![32, vocab]),
            ("position_embd.weight".into(), vec![32, 16]),
            ("output_norm.weight".into(), vec![32]),
            ("output_norm.bias".into(), vec![32]),
        ];
        for l in 0..2 {
            for (n, d) in [
                ("attn_norm.weight", vec![32]),
                ("attn_norm.bias", vec![32]),
                ("attn_qkv.weight", vec![32, 96]),
                ("attn_qkv.bias", vec![96]),
                ("attn_output.weight", vec![32, 32]),
                ("attn_output.bias", vec![32]),
                ("ffn_norm.weight", vec![32]),
                ("ffn_norm.bias", vec![32]),
                ("ffn_up.weight", vec![32, 64]),
                ("ffn_up.bias", vec![64]),
                ("ffn_down.weight", vec![64, 32]),
                ("ffn_down.bias", vec![32]),
            ] {
                v.push((format!("blk.{l}.{n}"), d));
            }
        }
        v
    };
    for (name, dims) in &expect {
        let t = g
            .tensor(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        assert_eq!(&t.dims, dims, "{name}");
    }
    // the output head is not stored separately: it is tied to the embedding (rule: a missing
    // output.weight means tied), or it has the embedding's shape transposed
    if let Some(h) = g.tensor("output.weight") {
        assert_eq!(h.dims, [32, vocab]);
    }
    assert_eq!(
        g.tensors.len(),
        expect.len() + usize::from(g.tensor("output.weight").is_some())
    );
}

#[test]
fn element_types_are_the_ones_each_precision_promises() {
    let f = fixture();
    for (path, matrix_type) in [(&f.f32, F32), (&f.q8, Q8_0), (&f.q4, Q4_0)] {
        let (_, g) = load(path);
        for t in &g.tensors {
            let is_matrix = t.dims.len() == 2 && !t.name.contains("position");
            let want = if is_matrix { matrix_type } else { F32 };
            assert_eq!(t.ggml_type, want, "{}: {}", path.display(), t.name);
        }
    }
}

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
