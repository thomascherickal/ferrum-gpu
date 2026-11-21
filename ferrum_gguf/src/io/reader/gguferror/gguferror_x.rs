/// A parse or decode failure.
#[derive(Debug, PartialEq, Eq)]
pub struct GgufError(pub String);

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gguf: {}", self.0)
    }
}

impl std::error::Error for GgufError {}

pub(crate) fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// u8
    U8(u8),
    /// i8
    I8(i8),
    /// u16
    U16(u16),
    /// i16
    I16(i16),
    /// u32
    U32(u32),
    /// i32
    I32(i32),
    /// f32
    F32(f32),
    /// bool
    Bool(bool),
    /// UTF-8 string
    Str(String),
    /// homogeneous array
    Arr(Vec<Value>),
    /// u64
    U64(u64),
    /// i64
    I64(i64),
    /// f64
    F64(f64),
}

/// One tensor's directory entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// GGML dims, innermost (contiguous) first.
    pub dims: Vec<u64>,
    /// ggml element type id.
    pub ggml_type: u32,
    /// Byte offset from the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Element count (checked).
    pub fn elements(&self) -> Result<usize, GgufError> {
        self.dims.iter().try_fold(1usize, |a, &d| {
            usize::try_from(d)
                .ok()
                .and_then(|d| a.checked_mul(d))
                .ok_or_else(|| err("tensor element count overflows"))
        })
    }
}
