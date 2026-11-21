use super::*;

/// [`inspect`] reading at most `max_header` bytes from the start.
pub fn inspect_with(path: &Path, max_header: u64) -> Result<Info, String> {
    let shown = path.display();
    let meta = std::fs::metadata(path).map_err(|e| format!("{shown}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("{shown}: not a file"));
    }
    let size = meta.len();
    let mut head = Vec::new();
    std::fs::File::open(path)
        .and_then(|f| f.take(max_header).read_to_end(&mut head))
        .map_err(|e| format!("{shown}: {e}"))?;
    let g = Gguf::parse_prefix(head, size).map_err(|e| {
        // a header longer than the prefix looks like a file that ends early
        let at_limit = size > max_header && e.0.contains("end of file");
        if at_limit {
            format!("{shown}: the header is larger than the {max_header} bytes read ({e})")
        } else {
            format!("{shown}: {e}")
        }
    })?;
    let mut kinds: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut parameters = 0u64;
    let mut tensor_bytes = 0u64;
    let mut rows = vec![];
    for t in &g.tensors {
        let elements = t.dims.iter().product::<u64>();
        let bytes = kinds::bytes_of(t.ggml_type, elements);
        parameters = parameters.saturating_add(elements);
        tensor_bytes = tensor_bytes.saturating_add(bytes.unwrap_or(0));
        let k = kinds.entry(kinds::name(t.ggml_type)).or_default();
        k.0 += 1;
        k.1 += bytes.unwrap_or(0);
        if rows.len() < MAX_ROWS {
            rows.push(TensorRow {
                name: t.name.clone(),
                dims: t.dims.clone(),
                kind: kinds::name(t.ggml_type),
                bytes,
            });
        }
    }
    let mut kinds: Vec<KindTotal> = kinds
        .into_iter()
        .map(|(kind, (tensors, bytes))| KindTotal {
            kind,
            tensors,
            bytes,
        })
        .collect();
    kinds.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.kind.cmp(&b.kind)));
    let text = |k: &str| match g.get(k) {
        Some(Value::Str(s)) => Some(s.clone()),
        _ => None,
    };
    Ok(Info {
        file_size: size,
        version: 3,
        architecture: text("general.architecture"),
        name: text("general.name"),
        n_tensors: g.tensors.len(),
        n_kv: g.kv.len(),
        parameters,
        tensor_bytes,
        kinds,
        metadata: g.kv.iter().map(|(k, v)| (k.clone(), show(v))).collect(),
        truncated: g.tensors.len() > rows.len(),
        tokenizer: tokenizer::tokenizer_info(&g),
        tensors: rows,
    })
}

/// The JSON the page reads.
pub fn to_json(i: &Info) -> Json {
    let num = |n: u64| Json::Num(n as f64);
    let opt = |s: &Option<String>| s.clone().map_or(Json::Null, Json::Str);
    Json::obj([
        ("file_size", num(i.file_size)),
        ("version", num(u64::from(i.version))),
        ("architecture", opt(&i.architecture)),
        ("name", opt(&i.name)),
        ("n_tensors", num(i.n_tensors as u64)),
        ("n_kv", num(i.n_kv as u64)),
        ("parameters", num(i.parameters)),
        ("tensor_bytes", num(i.tensor_bytes)),
        (
            "tokenizer",
            i.tokenizer.as_ref().map_or(Json::Null, |t| {
                Json::obj([
                    ("model", Json::Str(t.model.clone())),
                    ("tokens", num(t.tokens as u64)),
                    (
                        "chat_format",
                        t.chat_format.map_or(Json::Null, |f| Json::Str(f.into())),
                    ),
                ])
            }),
        ),
        (
            "kinds",
            Json::Arr(
                i.kinds
                    .iter()
                    .map(|k| {
                        Json::obj([
                            ("kind", Json::Str(k.kind.clone())),
                            ("tensors", num(k.tensors as u64)),
                            ("bytes", num(k.bytes)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "metadata",
            Json::Arr(
                i.metadata
                    .iter()
                    .map(|(k, v)| {
                        Json::obj([
                            ("key", Json::Str(k.clone())),
                            ("value", Json::Str(v.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "tensors",
            Json::Arr(
                i.tensors
                    .iter()
                    .map(|t| {
                        Json::obj([
                            ("name", Json::Str(t.name.clone())),
                            ("dims", Json::Arr(t.dims.iter().map(|d| num(*d)).collect())),
                            ("kind", Json::Str(t.kind.clone())),
                            ("bytes", t.bytes.map_or(Json::Null, num)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("truncated", Json::Bool(i.truncated)),
    ])
}
