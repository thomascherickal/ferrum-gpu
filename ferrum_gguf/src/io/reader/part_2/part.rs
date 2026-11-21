use super::*;

impl Gguf {
    /// Parse a GGUF v3 file.
    pub fn parse(bytes: Vec<u8>) -> Result<Gguf, GgufError> {
        let total = bytes.len() as u64;
        Gguf::parse_prefix(bytes, total)
    }

    /// Parse the header of a file that is `total` bytes long from its first bytes `bytes` (at
    /// least the metadata and tensor directory). Tensor extents are checked against `total`, so a
    /// truncated file is still refused; reading tensor data that lies beyond `bytes` is an error.
    /// Used to open huge files without reading them (token files).
    pub fn parse_prefix(bytes: Vec<u8>, total: u64) -> Result<Gguf, GgufError> {
        let mut c = Cur { b: &bytes, p: 0 };
        if c.u32()? != MAGIC {
            return Err(err("bad magic: not a GGUF file"));
        }
        let version = c.u32()?;
        if version != 3 {
            return Err(err(format!("unsupported GGUF version {version} (need 3)")));
        }
        let n_tensors = usize::try_from(c.u64()?).map_err(|_| err("tensor count overflows"))?;
        let n_kv = usize::try_from(c.u64()?).map_err(|_| err("metadata count overflows"))?;
        // Each entry takes at least 8 bytes, which bounds what the counts may claim.
        if n_kv > bytes.len() / 8 || n_tensors > bytes.len() / 8 {
            return Err(err("header counts exceed the file size"));
        }
        let mut kv = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            kv.push((key, c.value(ty, 0)?));
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            if nd == 0 || nd > 4 {
                return Err(err(format!(
                    "tensor '{name}' has {nd} dims (1 to 4 supported)"
                )));
            }
            let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>, _>>()?;
            let ggml_type = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }
        let alignment = match kv.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, Value::U32(a))) if *a > 0 => u64::from(*a),
            Some(_) => return Err(err("general.alignment must be a positive u32")),
            None => DEFAULT_ALIGNMENT,
        };
        let data_start = (c.p as u64).div_ceil(alignment) * alignment;
        let data_start = usize::try_from(data_start).map_err(|_| err("data offset overflows"))?;
        if data_start as u64 > total {
            return Err(err("file ends before the tensor data"));
        }
        let mut index = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            let size = nbytes(t.ggml_type, t.elements()?)?;
            let start = usize::try_from(t.offset)
                .ok()
                .and_then(|o| o.checked_add(data_start));
            if start
                .and_then(|s| s.checked_add(size))
                .is_none_or(|e| e as u64 > total)
            {
                return Err(err(format!(
                    "tensor '{}' extends past the end of the file",
                    t.name
                )));
            }
            if index.insert(t.name.clone(), i).is_some() {
                return Err(err(format!("duplicate tensor name '{}'", t.name)));
            }
        }
        Ok(Gguf {
            kv,
            tensors,
            index,
            data_start,
            bytes,
        })
    }

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
        usize::try_from(v).map_err(|_| err(format!("'{key}' is too large")))
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
