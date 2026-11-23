use super::*;

/// [`inspect`] reading at most `max_header` bytes from the start.
pub fn inspect_with(path: &Path, max_header: u64) -> Result<Info, String> {
    inspect_via(path, max_header, &|p| Ok(Box::new(std::fs::File::open(p)?)))
}

/// How a file is opened for reading.
pub(super) type Opener<'a> = &'a dyn Fn(&Path) -> std::io::Result<Box<dyn Read>>;

/// [`inspect_with`] reading through whatever `open` returns, so a failing source can be injected.
pub(super) fn inspect_via(path: &Path, max_header: u64, open: Opener) -> Result<Info, String> {
    let shown = path.display();
    let meta = std::fs::metadata(path).map_err(|e| format!("{shown}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("{shown}: not a file"));
    }
    let size = meta.len();
    let mut head = Vec::new();
    open(path)
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
