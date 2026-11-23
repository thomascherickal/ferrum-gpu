use super::*;

impl Gguf {
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
        let n_tensors = narrow(c.u64()?, "tensor count overflows")?;
        let n_kv = narrow(c.u64()?, "metadata count overflows")?;
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
        let data_start = narrow(data_start, "data offset overflows")?;
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
}
