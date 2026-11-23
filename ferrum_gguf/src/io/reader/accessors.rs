use super::*;

impl Gguf {
    /// A metadata value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// An unsigned integer metadata value (any integer width, non-negative).
    pub fn uint(&self, key: &str) -> Result<usize, GgufError> {
        let v = match self.get(key) {
            Some(Value::U8(v)) => u64::from(*v),
            Some(Value::U16(v)) => u64::from(*v),
            Some(Value::U32(v)) => u64::from(*v),
            Some(Value::U64(v)) => *v,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            Some(Value::I64(v)) if *v >= 0 => *v as u64,
            Some(_) => return Err(err(format!("'{key}' is not a non-negative integer"))),
            None => return Err(err(format!("missing key '{key}'"))),
        };
        narrow(v, format!("'{key}' is too large"))
    }

    /// An f32 metadata value.
    pub fn float(&self, key: &str) -> Result<f32, GgufError> {
        match self.get(key) {
            Some(Value::F32(v)) => Ok(*v),
            Some(_) => Err(err(format!("'{key}' is not an f32"))),
            None => Err(err(format!("missing key '{key}'"))),
        }
    }

    /// A string metadata value.
    pub fn string(&self, key: &str) -> Result<&str, GgufError> {
        match self.get(key) {
            Some(Value::Str(s)) => Ok(s),
            Some(_) => Err(err(format!("'{key}' is not a string"))),
            None => Err(err(format!("missing key '{key}'"))),
        }
    }

    /// A tensor's directory entry.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// The stored bytes of a tensor, undecoded (blocks of Q8_0 / Q4_0 stay packed).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| err(format!("missing tensor '{name}'")))?;
        let start = self.data_start + t.offset as usize; // inside the file, proven by parse
        let end = start + nbytes(t.ggml_type, t.elements()?)?;
        self.bytes.get(start..end).ok_or_else(|| {
            err(format!(
                "tensor '{name}' data was not loaded (header-only parse)"
            ))
        })
    }

    /// Where the tensor data starts in the file (the absolute offset of tensor offset 0).
    pub fn data_offset(&self) -> usize {
        self.data_start
    }

    /// Decode a tensor to f32 (row-major, GGML dim 0 contiguous).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| err(format!("missing tensor '{name}'")))?;
        decode(t.ggml_type, self.tensor_bytes(name)?, t.elements()?)
    }
}
