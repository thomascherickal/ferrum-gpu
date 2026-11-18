//! Looking inside a GGUF model without loading it: its header, metadata, tensors and size, read from
//! the start of the file only (`Gguf::parse_prefix`), so a model of any size can be inspected. The
//! reader is hostile-input safe (it is `ferrum_gguf`'s), and this module adds limits of its own.

pub mod kinds;
pub mod scan;
pub mod tokenizer;

use crate::json::Json;
use ferrum_gguf::{Gguf, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// The most bytes read from the start of a file to find its header.
pub const MAX_HEADER: u64 = 64 * 1024 * 1024;
/// Most tensor rows an inspection lists (the totals still count all of them).
pub const MAX_ROWS: usize = 2_000;
/// Longest metadata value shown, in characters.
pub const MAX_VALUE_CHARS: usize = 160;

/// One tensor of the directory.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRow {
    /// Its name.
    pub name: String,
    /// Its dimensions, ggml order (fastest first).
    pub dims: Vec<u64>,
    /// Its type name (`F32`, `Q8_0`, ...).
    pub kind: String,
    /// Bytes it takes in the file, if the type is known.
    pub bytes: Option<u64>,
}

/// How many tensors and bytes use one type.
#[derive(Clone, Debug, PartialEq)]
pub struct KindTotal {
    /// The type name.
    pub kind: String,
    /// Tensors of this type.
    pub tensors: usize,
    /// Their bytes (tensors of unknown size count as zero).
    pub bytes: u64,
}

/// What an inspection finds.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    /// Size of the file in bytes.
    pub file_size: u64,
    /// GGUF version (always 3 here).
    pub version: u32,
    /// `general.architecture`, if present.
    pub architecture: Option<String>,
    /// `general.name`, if present.
    pub name: Option<String>,
    /// Tensors in the directory.
    pub n_tensors: usize,
    /// Metadata entries.
    pub n_kv: usize,
    /// Total weights: the product of the dimensions of every tensor.
    pub parameters: u64,
    /// Bytes of tensor data.
    pub tensor_bytes: u64,
    /// Totals per tensor type, largest first.
    pub kinds: Vec<KindTotal>,
    /// Metadata as `(key, shown value)`, in file order.
    pub metadata: Vec<(String, String)>,
    /// Tensor rows, at most [`MAX_ROWS`].
    pub tensors: Vec<TensorRow>,
    /// There were more tensors than rows listed.
    pub truncated: bool,
    /// The tokenizer the file carries, if any.
    pub tokenizer: Option<tokenizer::TokInfo>,
}

/// A metadata value as text: strings quoted and cut, arrays summarised.
pub fn show(v: &Value) -> String {
    let cut = |s: String| {
        if s.chars().count() > MAX_VALUE_CHARS {
            format!("{}…", s.chars().take(MAX_VALUE_CHARS).collect::<String>())
        } else {
            s
        }
    };
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
        Value::Str(s) => cut(format!("\"{s}\"")),
        Value::Arr(a) => cut(format!(
            "[{} items] {}",
            a.len(),
            a.iter().take(6).map(show).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Inspect the GGUF file at `path` (reading at most [`MAX_HEADER`] bytes of it).
pub fn inspect(path: &Path) -> Result<Info, String> {
    inspect_with(path, MAX_HEADER)
}

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

#[cfg(test)]
mod tests;
