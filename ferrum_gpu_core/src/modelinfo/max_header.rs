use super::*;

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
