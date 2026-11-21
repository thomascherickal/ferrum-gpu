use super::*;

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
